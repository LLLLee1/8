import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import os
from lightfm import LightFM
from lightfm.data import Dataset
import scipy.sparse as sparse
import subprocess
import sys

def check_surprise_installation():
    """深度检查surprise库安装状态"""
    try:
        import surprise
        st.success(f"surprise库版本: {surprise.__version__}")
        
        # 检查核心模块是否编译成功
        from surprise import SVD
        st.success("surprise核心模块加载成功")
        
        # 测试基本功能
        dummy_data = [(1, 1, 5.0), (1, 2, 3.0)]
        reader = surprise.Reader(rating_scale=(1, 5))
        data = surprise.Dataset.load_from_list(dummy_data, reader)
        trainset = data.build_full_trainset()
        algo = SVD()
        algo.fit(trainset)
        st.success("surprise功能测试通过")
        return True
        
    except Exception as e:
        st.error(f"surprise库加载失败: {str(e)}")
        
        # 捕获编译错误细节
        st.info("编译错误详情:")
        try:
            output = subprocess.check_output(
                [sys.executable, "-c", "import surprise"],
                stderr=subprocess.STDOUT,
                text=True
            )
            st.code(output)
        except subprocess.CalledProcessError as cpe:
            st.code(cpe.output)
        
        return False

# 检查是否在Streamlit环境中运行
def is_running_in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        return False

# 根据环境使用不同的缓存装饰器，避免ScriptRunContext警告
if is_running_in_streamlit():
    cache_resource = st.cache_resource
    cache_data = st.cache_data
else:
    def cache_resource(func):
        return func
    def cache_data(func):
        return func

# 确保会话状态初始化的函数
def initialize_session_state():
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = False
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'selected_jokes' not in st.session_state:
        st.session_state.selected_jokes = []
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = {}
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'recommendation_ratings' not in st.session_state:
        st.session_state.recommendation_ratings = {}
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'ratings_df' not in st.session_state:
        st.session_state.ratings_df = None
    if 'jokes_df' not in st.session_state:
        st.session_state.jokes_df = None

# 修改模型训练和预测逻辑
@cache_resource
def load_trained_model():
    """加载训练好的模型（适配lightfm）"""
    model_path = "output/trained_model.pkl"
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
            # 提取lightfm模型和数据
            model = model_data['model']
            dataset = model_data['dataset']
            
        st.session_state.model_loaded = True
        st.success(f"模型加载成功，类型: {type(model)}")
        return model, dataset
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None, None

# 修改推荐生成函数
def generate_recommendations_with_model(user_ratings, model, dataset, jokes_df):
    """使用lightfm模型生成推荐"""
    if not user_ratings:
        st.warning("请先为笑话评分")
        return []
    
    # 获取用户和物品映射
    user_id_map, _, item_id_map, _ = dataset.mapping()
    
    # 构建用户评分矩阵
    user_item_matrix = dataset.build_interactions(
        [(0, item_id_map[joke_id], rating) for joke_id, rating in user_ratings.items() if joke_id in item_id_map]
    )[0]
    
    # 生成推荐
    all_joke_ids = list(item_id_map.keys())
    scores = model.predict(
        user_ids=0,  # 新用户ID为0
        item_ids=[item_id_map[joke_id] for joke_id in all_joke_ids]
    )
    
    # 排序并获取Top-N推荐
    top_indices = scores.argsort()[::-1][:5]
    
    # 获取推荐笑话详情
    recommendations_with_content = []
    for idx in top_indices:
        joke_id = all_joke_ids[idx]
        joke_content = jokes_df.loc[joke_id, 'joke']
        recommendations_with_content.append({
            'joke_id': joke_id,
            'predicted_rating': scores[idx],
            'content': joke_content
        })
    
    return recommendations_with_content

# 计算满意度
def calculate_satisfaction(rec_ratings):
    """计算用户满意度"""
    if not rec_ratings:
        return 0
    
    # 评分范围是-10到10
    min_rating, max_rating = -10, 10
    
    # 计算平均评分
    avg_rating = sum(rec_ratings.values()) / len(rec_ratings)
    
    # 归一化到0-100%
    satisfaction = ((avg_rating - min_rating) / (max_rating - min_rating)) * 100
    
    return satisfaction

# 显示随机笑话
def display_random_jokes(jokes_df):
    """显示随机笑话并收集评分"""
    st.header("请为以下笑话评分")
    
    # 如果还没有选择笑话，随机选择3个
    if not st.session_state.selected_jokes:
        joke_ids = jokes_df.index.tolist()
        st.session_state.selected_jokes = random.sample(joke_ids, 3)
    
    # 创建3列布局
    cols = st.columns(3)
    
    for i, joke_id in enumerate(st.session_state.selected_jokes):
        with cols[i]:
            st.subheader(f"笑话 {i+1}")
            st.write(jokes_df.loc[joke_id, 'joke'])
            
            # 评分组件
            rating = st.slider(
                f"为这个笑话评分 (-10到10)",
                min_value=-10,
                max_value=10,
                value=0,
                step=1,
                key=f"rating_{joke_id}"
            )
            
            # 保存评分
            if st.button(f"提交笑话 {i+1} 的评分", key=f"submit_{joke_id}"):
                st.session_state.user_ratings[joke_id] = rating
                st.success(f"已记录评分: {rating}")
    
    # 显示已评分的笑话数量
    rated_count = len(st.session_state.user_ratings)
    st.write(f"已评分: {rated_count}/3")
    
    # 如果已经评了3个笑话，允许进入下一步
    if rated_count >= 3:
        if st.button("获取个性化推荐"):
            st.session_state.current_step = 3

# 显示推荐结果
def display_recommendations(model, jokes_df):
    """显示推荐结果"""
    st.header("为您推荐的笑话")
    
    # 生成推荐
    if not st.session_state.recommendations:
        with st.spinner("正在生成推荐..."):
            st.session_state.recommendations = generate_recommendations_with_model(
                st.session_state.user_ratings, 
                model,
                jokes_df
            )
    
    # 显示推荐
    for i, rec in enumerate(st.session_state.recommendations):
        st.subheader(f"推荐 {i+1}")
        st.write(f"预测评分: {rec['predicted_rating']:.2f}")
        st.write(rec['content'])
        
        # 为推荐评分
        rating = st.slider(
            f"为这个推荐评分 (-10到10)",
            min_value=-10,
            max_value=10,
            value=0,
            step=1,
            key=f"rec_rating_{rec['joke_id']}"
        )
        
        if st.button(f"提交推荐 {i+1} 的评分", key=f"rec_submit_{rec['joke_id']}"):
            st.session_state.recommendation_ratings[rec['joke_id']] = rating
            st.success(f"已记录评分: {rating}")
    
    # 显示已评分数
    rated_count = len(st.session_state.recommendation_ratings)
    st.write(f"已为推荐评分: {rated_count}/5")
    
    # 如果已经评了5个推荐，允许计算满意度
    if rated_count >= 5:
        if st.button("计算推荐满意度"):
            st.session_state.current_step = 4

# 显示满意度计算结果
def display_satisfaction():
    """显示满意度计算结果"""
    st.header("推荐满意度计算")
    
    # 计算满意度
    satisfaction = calculate_satisfaction(st.session_state.recommendation_ratings)
    
    # 显示结果
    st.subheader(f"您的推荐满意度: {satisfaction:.2f}%")
    
    # 满意度解释
    if satisfaction >= 80:
        st.success("🎉 非常满意！推荐系统表现出色，为您提供了高质量的笑话推荐。")
    elif satisfaction >= 60:
        st.info("😊 比较满意！推荐系统基本符合您的口味，继续探索更多笑话吧。")
    elif satisfaction >= 40:
        st.warning("😐 一般满意！推荐系统还有改进空间，我们会努力理解您的偏好。")
    else:
        st.error("😔 不太满意！很抱歉推荐没有达到您的期望，我们会继续优化。")
    
    # 显示评分明细
    st.subheader("评分明细")
    for joke_id, rating in st.session_state.recommendation_ratings.items():
        st.write(f"笑话ID {joke_id}: 评分 {rating}")
    
    # 重新开始按钮
    if st.button("重新开始"):
        # 重置会话状态
        st.session_state.selected_jokes = []
        st.session_state.user_ratings = {}
        st.session_state.recommendations = []
        st.session_state.recommendation_ratings = {}
        st.session_state.current_step = 1
        st.success("已重置所有评分，您可以重新开始使用系统。")

# 主应用
def main():
    st.set_page_config(page_title="笑话推荐系统", layout="wide")
    st.title("基于协同过滤的笑话推荐系统")
    
    # 初始化会话状态
    initialize_session_state()
    
    # 侧边栏
    with st.sidebar:
        st.header("应用信息")
        st.write("本系统使用协同过滤算法，根据您的笑话评分提供个性化推荐。")
        st.markdown("---")
        
        # 显示模型信息
        if st.session_state.model_loaded:
            st.subheader("模型信息")
            st.write(f"模型类型: {detect_model_type(st.session_state.model)}")
            st.write(f"训练数据: {len(st.session_state.ratings_df)} 条评分记录")
            st.write(f"笑话数量: {len(st.session_state.jokes_df)} 个")
        
        st.markdown("---")
        st.write("© 2025 笑话推荐系统")
    
    # 检查应用是否已初始化
    if not st.session_state.app_initialized:
        st.info("请先加载模型和数据...")
        
        # 文件上传
        model_file = st.file_uploader("上传模型文件 (.pkl)", type=["pkl"])
        ratings_file = st.file_uploader("上传评分数据 (.csv)", type=["csv"])
        jokes_file = st.file_uploader("上传笑话文本数据 (.xlsx)", type=["xlsx"])
        
        if model_file and ratings_file and jokes_file:
            # 保存上传的文件
            with open("model.pkl", "wb") as f:
                f.write(model_file.getbuffer())
            
            with open("ratings.csv", "wb") as f:
                f.write(ratings_file.getbuffer())
            
            with open("jokes.xlsx", "wb") as f:
                f.write(jokes_file.getbuffer())
            
            # 加载模型和数据
            with st.spinner("正在加载模型和数据..."):
                model = load_trained_model("model.pkl")
                ratings_df, jokes_df = load_experiment_data("ratings.csv", "jokes.xlsx")
                
                if model and ratings_df is not None and jokes_df is not None:
                    # 保存到会话状态
                    st.session_state.model = model
                    st.session_state.ratings_df = ratings_df
                    st.session_state.jokes_df = jokes_df
                    st.session_state.app_initialized = True
                    
                    st.success("模型和数据加载成功！")
                    st.balloons()
    
    # 主应用流程
    if st.session_state.app_initialized:
        # 步骤指示器
        steps = ["欢迎", "笑话评分", "推荐结果", "满意度"]
        current_step_index = st.session_state.current_step - 1
        
        # 显示步骤导航
        st.subheader(f"步骤 {current_step_index + 1}/{len(steps)}: {steps[current_step_index]}")
        
        # 根据当前步骤显示相应内容
        if current_step_index == 0:
            st.write("欢迎使用笑话推荐系统！本系统会根据您对笑话的评分，为您推荐个性化的笑话。")
            st.write("请点击下方按钮开始为笑话评分。")
            
            if st.button("开始评分"):
                st.session_state.current_step = 1
        
        elif current_step_index == 1:
            display_random_jokes(st.session_state.jokes_df)
        
        elif current_step_index == 2:
            display_recommendations(st.session_state.model, st.session_state.jokes_df)
        
        elif current_step_index == 3:
            display_satisfaction()

if __name__ == "__main__":
    main()
