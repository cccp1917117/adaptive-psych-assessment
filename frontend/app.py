import streamlit as st
import requests

# 1. 后端 FastAPI 地址
BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI 自适应测评系统", page_icon="🧠", layout="centered")

st.title("🧠 心理健康自适应测评系统")
st.write("基于多阶段自适应测试（MST）技术，动态调整后续题目。")
st.divider()

# 2. 初始化全局状态机
if "started" not in st.session_state:
    st.session_state.started = False
    st.session_state.is_finished = False
    st.session_state.current_response = None  # 存储后端当前返回的整个大 JSON
    st.session_state.all_answers = []         # 🌟 核心：保存历史至今所有的 [{"question_id": "...", "score": X}, ...]

# SCL-90 标准 5 级评分映射
OPTIONS = {
    "0 - 从无 (None)": 0,
    "1 - 轻度 (Mild)": 1,
    "2 - 中度 (Moderate)": 2,
    "3 - 偏重 (Severe)": 3,
    "4 - 严重 (Extreme)": 4
}

# ================= 场景 A：欢迎与启动页面 =================
if not st.session_state.started and not st.session_state.is_finished:
    st.subheader("开始测评")
    st.info("💡 说明：请根据自己【最近一周内】的实际感觉进行选择。")
    
    if st.button("🚀 开始 AI 测评", type="primary"):
        try:
            # 呼叫后端的 GET /start
            response = requests.get(f"{BACKEND_URL}/start") 
            
            if response.status_code == 200:
                # 存入第一轮数据（包含 stage 和 questions 列表）
                st.session_state.current_response = response.json()
                st.session_state.all_answers = [] # 清空历史
                st.session_state.started = True
                st.rerun()
            else:
                st.error(f"后端连接异常，状态码: {response.status_code}")
        except Exception as e:
            st.error(f"无法连接到后端服务器，请确认后端已运行 uvicorn。错误: {e}")

# ================= 场景 B：自适应循环答题页面 =================
elif st.session_state.started and not st.session_state.is_finished:
    current_data = st.session_state.current_response
    
    # 第一轮后端返回叫 "questions"；后续轮次返回叫 "next_questions"
    questions_list = current_data.get("next_questions") if "next_questions" in current_data else current_data.get("questions", [])
    
    # 实时因子分展示（展示科研成果亮点）
    current_scores = current_data.get("scores")
    if current_scores:
        st.subheader("📊 AI 实时因子评分 (实时动态更新)")
        st.caption("后端算法基于您目前的所有回答计算出的各维度均分：")
        st.json(current_scores)
        st.divider()

    # 🛑 核心终止判定：如果后端给的题目列表变空了，说明算法收敛，测试结束！
    if not questions_list:
        st.session_state.started = False
        st.session_state.is_finished = True
        st.session_state.rerun()

    st.subheader("📋 请回答以下自适应调整题目：")
    
    # 用来临时收集当前页面上这几道题的填答结果
    current_page_answers = {}
    
    # 动态渲染当前轮次的题目
    for q in questions_list:
        q_id = q.get("id")
        q_text = q.get("text")
        q_dim = q.get("dimension")
        
        st.markdown(f"**题目编号: {q_id}** *(考察维度: {q_dim})*")
        # 为每道题生成一个下拉选择框
        choice = st.selectbox(f"{q_text}", list(OPTIONS.keys()), key=f"select_{q_id}")
        # 记录分数
        current_page_answers[q_id] = OPTIONS[choice]
        st.write("")
        
    st.divider()
    
    if st.button("➡️ 提交当前回答", type="primary"):
        # 1. 把当前页面的答案，转化成后端要求的格式：{"question_id": "...", "score": ...}
        formatted_current_answers = [
            {"question_id": qid, "score": val} for qid, val in current_page_answers.items()
        ]
        
        # 2. 🌟 关键：把当前轮次的答案，追加合并到前端的总历史答案库里
        st.session_state.all_answers.extend(formatted_current_answers)
        
        # 3. 按照后端 AnswerSet 模型的严格要求打包 Payload
        payload = {
            "answers": st.session_state.all_answers # 发送历史至今的所有答案！
        }
        
        try:
            # 呼叫后端的 POST /next
            response = requests.post(f"{BACKEND_URL}/next", json=payload)
            
            if response.status_code == 200:
                next_data = response.json()
                
                # 检查下一轮还有没有题
                if not next_data.get("next_questions"):
                    # 没题了，通关！
                    st.session_state.started = False
                    st.session_state.is_finished = True
                else:
                    # 还有题，把后端的回信整体更新到状态机，进入下一轮循环
                    st.session_state.current_response = next_data
                st.rerun()
                
            elif response.status_code == 422:
                st.error("❌ 前端触发 422 数据校验错误！请检查发送的 JSON 格式。")
                st.json(payload)
            else:
                st.error(f"❌ 后端逻辑崩溃 (500)，请查看后端的 Python 控制台日志。")
        except Exception as e:
            st.error(f"网络请求失败: {e}")

# ================= 场景 C：最终测评报告页面 =================
elif st.session_state.is_finished:
    st.balloons() # 庆祝特效
    st.success("🎉 自适应测评顺利完成！AI 算法已成功收敛。")
    
    st.header("🏁 最终全维度心理健康因子报告")
    
    # 拿到最后一轮后端计算出来的完整 scores
    final_scores = st.session_state.current_response.get("scores") if st.session_state.current_response else None
    
    if final_scores:
        # Streamlit 杀手级功能：直接把 Python 字典画成精美的柱状图
        st.bar_chart(final_scores)
        
        st.write("📊 因子均分详情：")
        st.json(final_scores)
    else:
        st.warning("未能获取到最终的评分数据。")
        
    if st.button("🔄 重新开始测评"):
        st.session_state.clear()
        st.rerun()