import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import os

# 检查是否在Streamlit环境中运行
def is_running_in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        return False

# 根据环境使用不同的缓存装饰器
if is_running_in_streamlit():
    cache_resource = st.cache_resource
    cache_data = st.cache_data
else:
    def cache_resource(func):
        return func
    def cache_data(func):
        return func

# 会话状态初始化
def initialize_session_state():
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = False
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
    if 'ratings_df' not in st.session_state:
        st.session_state.ratings_df = None
    if 'jokes_df' not in st.session_state:
        st.session_state.jokes_df = None

# 加载数据
@cache_data
def load_experiment_data():
    """加载实验数据（纯Pandas处理）"""
    ratings_path = "output/processed_ratings.csv"
    jokes_path = "output/Dataset4JokeSet.xlsx"
    try:
        ratings_df = pd.read_csv(ratings_path)
        jokes_df = pd.read_excel(jokes_path)
        jokes_df.set_index('joke_id', inplace=True)
        return ratings_df, jokes_df
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return None, None

# 构建用户-笑话评分矩阵
def build_rating_matrix(ratings_df, jokes_df):
    """使用Pandas构建评分矩阵"""
    if ratings_df is None or jokes_df is None:
        return None, None, None
    
    # 提取所有用户和笑话ID
    user_ids = ratings_df['user_id'].unique()
    joke_ids = jokes_df.index.tolist()
    
    # 创建评分矩阵
    rating_matrix = pd.DataFrame(0, index=user_ids, columns=joke_ids)
    
    # 填充评分数据
    for _, row in ratings_df.iterrows():
        if row['joke_id'] in joke_ids:
            rating_matrix.loc[row['user_id'], row['joke_id']] = row['rating']
    
    return rating_matrix, user_ids, joke_ids

# 计算用户相似度（余弦相似度）
def calculate_user_similarity(rating_matrix):
    """纯NumPy计算用户相似度"""
    if rating_matrix is None:
        return None
    
    # 转换为NumPy数组
    matrix = rating_matrix.to_numpy()
    
    # 计算余弦相似度
    similarities = np.zeros((len(matrix), len(matrix)))
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            # 过滤掉全零行（无评分的用户）
            if np.linalg.norm(matrix[i]) > 0 and np.linalg.norm(matrix[j]) > 0:
                similarity = np.dot(matrix[i], matrix[j]) / (np.linalg.norm(matrix[i]) * np.linalg.norm(matrix[j]))
                similarities[i, j] = similarity
                similarities[j, i] = similarity
    
    return similarities

# 生成推荐
def generate_recommendations(user_ratings, rating_matrix, user_ids, joke_ids, jokes_df, top_n=5):
    """纯Python生成推荐"""
    if not user_ratings or rating_matrix is None:
        return []
    
    # 构建新用户评分向量
    new_user_vector = np.zeros(len(joke_ids))
    for joke_id, rating in user_ratings.items():
        if joke_id in joke_ids:
            joke_idx = joke_ids.index(joke_id)
            new_user_vector[joke_idx] = rating
    
    # 计算新用户与已有用户的相似度
    similarities = np.zeros(len(user_ids))
    for i, user_id in enumerate(user_ids):
        existing_user_vector = rating_matrix.loc[user_id].values
        # 过滤掉无评分的用户
        if np.linalg.norm(existing_user_vector) > 0 and np.linalg.norm(new_user_vector) > 0:
            similarity = np.dot(new_user_vector, existing_user_vector) / (np.linalg.norm(new_user_vector) * np.linalg.norm(existing_user_vector))
            similarities[i] = similarity
    
    # 找到最相似的K个用户
    k = min(5, len(user_ids))
    similar_user_indices = np.argsort(-similarities)[:k]
    similar_user_weights = similarities[similar_user_indices]
    
    # 基于相似用户的评分加权平均
    recommendation_scores = np.zeros(len(joke_ids))
    for i, idx in enumerate(similar_user_indices):
        recommendation_scores += similar_user_weights[i] * rating_matrix.loc[user_ids[idx]].values
    
    # 过滤掉已评分的笑话
    for joke_id in user_ratings:
        if joke_id in joke_ids:
            joke_idx = joke_ids.index(joke_id)
            recommendation_scores[joke_idx] = -np.inf
    
    # 获取Top-N推荐
    top_indices = np.argsort(-recommendation_scores)[:top_n]
    
    # 转换为笑话内容
    recommendations = []
    for idx in top_indices:
        joke_id = joke_ids[idx]
        recommendations.append({
            'joke_id': joke_id,
            'predicted_rating': recommendation_scores[idx],
            'content': jokes_df.loc[joke_id, 'joke']
        })
    
    return recommendations

# 计算满意度
def calculate_satisfaction(rec_ratings):
    """计算用户满意度"""
    if not rec_ratings:
        return 0
    avg_rating = sum(rec_ratings.values()) / len(rec_ratings)
    satisfaction = ((avg_rating - (-10)) / (10 - (-10))) * 100
    return satisfaction

# 显示随机笑话
def display_random_jokes(jokes_df):
    """显示随机笑话并收集评分"""
    st.header("请为以下笑话评分")
    if not st.session_state.selected_jokes:
        joke_ids = jokes_df.index.tolist()
        st.session_state.selected_jokes = random.sample(joke_ids, 3)
    
    cols = st.columns(3)
    for i, joke_id in enumerate(st.session_state.selected_jokes):
        with cols[i]:
            st.subheader(f"笑话 {i+1}")
            st.write(jokes_df.loc[joke_id, 'joke'])
            rating = st.slider(f"评分 (-10到10)", -10, 10, 0, key=f"rating_{joke_id}")
            if st.button(f"提交评分", key=f"submit_{joke_id}"):
                st.session_state.user_ratings[joke_id] = rating
                st.success(f"已记录评分: {rating}")
    
    rated_count = len(st.session_state.user_ratings)
    st.write(f"已评分: {rated_count}/3")
    if rated_count >= 3 and st.button("获取推荐"):
        st.session_state.current_step = 3

# 显示推荐结果
def display_recommendations(rating_matrix, user_ids, joke_ids, jokes_df):
    """显示推荐结果"""
    st.header("为您推荐的笑话")
    if not st.session_state.recommendations:
        with st.spinner("生成推荐中..."):
            st.session_state.recommendations = generate_recommendations(
                st.session_state.user_ratings, 
                rating_matrix, 
                user_ids, 
                joke_ids, 
                jokes_df
            )
    
    for i, rec in enumerate(st.session_state.recommendations):
        st.subheader(f"推荐 {i+1}")
        st.write(f"预测评分: {rec['predicted_rating']:.2f}")
        st.write(rec['content'])
        rating = st.slider(f"为推荐评分 (-10到10)", -10, 10, 0, key=f"rec_rating_{rec['joke_id']}")
        if st.button(f"提交评分", key=f"rec_submit_{rec['joke_id']}"):
            st.session_state.recommendation_ratings[rec['joke_id']] = rating
            st.success(f"已记录评分: {rating}")
    
    rated_count = len(st.session_state.recommendation_ratings)
    st.write(f"已评分: {rated_count}/5")
    if rated_count >= 5 and st.button("计算满意度"):
        st.session_state.current_step = 4

# 显示满意度
def display_satisfaction():
    """显示满意度计算结果"""
    st.header("推荐满意度")
    satisfaction = calculate_satisfaction(st.session_state.recommendation_ratings)
    st.subheader(f"满意度: {satisfaction:.2f}%")
    
    if satisfaction >= 80:
        st.success("🎉 非常满意！")
    elif satisfaction >= 60:
        st.info("😊 比较满意！")
    elif satisfaction >= 40:
        st.warning("😐 一般满意！")
    else:
        st.error("😔 不太满意！")
    
    st.subheader("评分明细")
    for joke_id, rating in st.session_state.recommendation_ratings.items():
        st.write(f"笑话 {joke_id}: 评分 {rating}")
    
    if st.button("重新开始"):
        st.session_state.selected_jokes = []
        st.session_state.user_ratings = {}
        st.session_state.recommendations = []
        st.session_state.recommendation_ratings = {}
        st.session_state.current_step = 1
        st.success("已重置所有评分")

# 主应用
def main():
    st.set_page_config(page_title="笑话推荐系统", layout="wide")
    st.title("基于纯Python协同过滤的笑话推荐系统")
    
    # 初始化会话状态
    initialize_session_state()
    
    # 侧边栏
    with st.sidebar:
        st.header("系统信息")
        st.write("纯Python实现，无任何推荐库依赖")
        st.markdown("---")
        if st.session_state.ratings_df is not None and st.session_state.jokes_df is not None:
            st.write(f"训练数据: {len(st.session_state.ratings_df)} 条评分")
            st.write(f"笑话数量: {len(st.session_state.jokes_df)} 个")
        st.markdown("---")
        st.write("© 2025 纯Python推荐系统")
    
    # 加载数据
    if not st.session_state.app_initialized:
        st.info("请先加载数据...")
        ratings_file = st.file_uploader("上传评分数据 (.csv)", type=["csv"])
        jokes_file = st.file_uploader("上传笑话文本 (.xlsx)", type=["xlsx"])
        
        if ratings_file and jokes_file:
            with open("ratings.csv", "wb") as f:
                f.write(ratings_file.getbuffer())
            with open("jokes.xlsx", "wb") as f:
                f.write(jokes_file.getbuffer())
            
            with st.spinner("加载数据中..."):
                ratings_df, jokes_df = load_experiment_data()
                if ratings_df is not None and jokes_df is not None:
                    st.session_state.ratings_df = ratings_df
                    st.session_state.jokes_df = jokes_df
                    st.session_state.app_initialized = True
                    st.success("数据加载成功！")
    
    # 主流程
    if st.session_state.app_initialized:
        steps = ["欢迎", "笑话评分", "推荐结果", "满意度"]
        current_step = st.session_state.current_step - 1
        
        st.subheader(f"步骤 {current_step + 1}/{len(steps)}: {steps[current_step]}")
        
        if current_step == 0:
            st.write("欢迎使用纯Python实现的笑话推荐系统！")
            st.write("系统基于用户评分相似度为您推荐笑话，无需任何编译依赖。")
            if st.button("开始评分"):
                st.session_state.current_step = 1
        
        elif current_step == 1:
            display_random_jokes(st.session_state.jokes_df)
        
        elif current_step == 2:
            # 构建评分矩阵和相似度
            rating_matrix, user_ids, joke_ids = build_rating_matrix(
                st.session_state.ratings_df, 
                st.session_state.jokes_df
            )
            if rating_matrix is not None:
                display_recommendations(rating_matrix, user_ids, joke_ids, st.session_state.jokes_df)
        
        elif current_step == 3:
            display_satisfaction()

if __name__ == "__main__":
    main()
