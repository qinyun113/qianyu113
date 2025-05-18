import os
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid
import joblib
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# 确保上传文件夹存在
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 模拟风险数据
risk_data = [
    {
        "id": "R-001",
        "project_name": "市场拓展项目",
        "risk_level": "高风险",
        "risk_type": "市场风险",
        "probability": "80%",
        "impact": "严重",
        "risk_score": 78.5,
        "status": "处理中",
        "create_time": "2023-12-25"
    },
    {
        "id": "R-002",
        "project_name": "新产品研发",
        "risk_level": "中风险",
        "risk_type": "技术风险",
        "probability": "60%",
        "impact": "中等",
        "risk_score": 58.2,
        "status": "未处理",
        "create_time": "2023-12-26"
    },
    {
        "id": "R-003",
        "project_name": "供应链优化",
        "risk_level": "低风险",
        "risk_type": "操作风险",
        "probability": "20%",
        "impact": "轻微",
        "risk_score": 18.7,
        "status": "已处理",
        "create_time": "2023-12-20"
    },
    {
        "id": "R-004",
        "project_name": "海外市场投资",
        "risk_level": "高风险",
        "risk_type": "政治风险",
        "probability": "90%",
        "impact": "严重",
        "risk_score": 87.3,
        "status": "处理中",
        "create_time": "2023-12-22"
    },
    {
        "id": "R-005",
        "project_name": "并购项目",
        "risk_level": "中风险",
        "risk_type": "财务风险",
        "probability": "50%",
        "impact": "中等",
        "risk_score": 48.9,
        "status": "未处理",
        "create_time": "2023-12-28"
    }
]

# 模拟上传历史
upload_history = [
    {
        "id": "U-001",
        "filename": "risk_data_2023.csv",
        "project_name": "市场拓展项目",
        "data_type": "历史风险数据",
        "upload_time": "2023-12-31 14:30:25",
        "status": "成功",
        "size": "245 KB"
    },
    {
        "id": "U-002",
        "filename": "market_data.xlsx",
        "project_name": "新产品研发",
        "data_type": "市场数据",
        "upload_time": "2023-12-30 10:15:42",
        "status": "成功",
        "size": "1.2 MB"
    }
]

# 检查文件扩展名是否允许
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 首页路由
@app.route('/')
def index():
    return render_template('index.html')

# 获取风险数据
@app.route('/api/risks', methods=['GET'])
def get_risks():
    # 支持分页和筛选
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    sort_by = request.args.get('sort_by', 'create_time')
    sort_order = request.args.get('sort_order', 'desc')
    project_name = request.args.get('project_name', '')
    risk_level = request.args.get('risk_level', '')
    risk_type = request.args.get('risk_type', '')
    status = request.args.get('status', '')
    
    # 筛选数据
    filtered_data = risk_data
    if project_name:
        filtered_data = [risk for risk in filtered_data if project_name.lower() in risk['project_name'].lower()]
    if risk_level:
        filtered_data = [risk for risk in filtered_data if risk['risk_level'] == risk_level]
    if risk_type:
        filtered_data = [risk for risk in filtered_data if risk['risk_type'] == risk_type]
    if status:
        filtered_data = [risk for risk in filtered_data if risk['status'] == status]
    
    # 排序数据
    if sort_order == 'desc':
        filtered_data.sort(key=lambda x: x[sort_by], reverse=True)
    else:
        filtered_data.sort(key=lambda x: x[sort_by])
    
    # 分页
    start = (page - 1) * per_page
    end = start + per_page
    paginated_data = filtered_data[start:end]
    
    return jsonify({
        'data': paginated_data,
        'total': len(filtered_data),
        'page': page,
        'per_page': per_page,
        'pages': (len(filtered_data) + per_page - 1) // per_page
    })

# 获取风险统计数据
@app.route('/api/risk_stats', methods=['GET'])
def get_risk_stats():
    total_risks = len(risk_data)
    high_risk = len([risk for risk in risk_data if risk['risk_level'] == '高风险'])
    medium_risk = len([risk for risk in risk_data if risk['risk_level'] == '中风险'])
    low_risk = len([risk for risk in risk_data if risk['risk_level'] == '低风险'])
    
    # 计算风险趋势数据
    trend_data = {
        'labels': ['周一', '周二', '周三', '周四', '周五', '周六', '周日'],
        'risk_index': [65, 59, 80, 81, 56, 55, 68],
        'high_risk_count': [12, 19, 15, 11, 13, 10, 12]
    }
    
    # 计算风险类型分布
    risk_types = {}
    for risk in risk_data:
        risk_types[risk['risk_type']] = risk_types.get(risk['risk_type'], 0) + 1
    
    return jsonify({
        'total_risks': total_risks,
        'high_risk': high_risk,
        'medium_risk': medium_risk,
        'low_risk': low_risk,
        'risk_trend': trend_data,
        'risk_types': risk_types
    })

# 上传文件
@app.route('/api/upload', methods=['POST'])
def upload_file():
    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    project_name = request.form.get('project_name')
    data_type = request.form.get('data_type')
    
    # 如果用户没有选择文件，浏览器可能会提交一个空的文件
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    # 检查文件类型
    if file and allowed_file(file.filename):
        # 生成唯一文件名
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # 保存文件
        try:
            file.save(file_path)
            
            # 记录上传历史
            file_size = os.path.getsize(file_path)
            upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            upload_entry = {
                'id': f"U-{len(upload_history) + 1:03d}",
                'filename': filename,
                'project_name': project_name,
                'data_type': data_type,
                'upload_time': upload_time,
                'status': '成功',
                'size': f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB",
                'file_path': unique_filename
            }
            
            upload_history.append(upload_entry)
            
            # 处理上传的文件（这里可以添加数据分析逻辑）
            if data_type == '历史风险数据':
                process_risk_data(file_path)
            
            return jsonify({
                'message': '文件上传成功',
                'upload_data': upload_entry
            }), 201
        
        except Exception as e:
            return jsonify({'error': f'上传失败: {str(e)}'}), 500
    
    return jsonify({'error': '不支持的文件类型'}), 400

# 获取上传历史
@app.route('/api/upload_history', methods=['GET'])
def get_upload_history():
    # 支持分页
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    # 排序（按上传时间倒序）
    sorted_history = sorted(upload_history, key=lambda x: x['upload_time'], reverse=True)
    
    # 分页
    start = (page - 1) * per_page
    end = start + per_page
    paginated_data = sorted_history[start:end]
    
    return jsonify({
        'data': paginated_data,
        'total': len(sorted_history),
        'page': page,
        'per_page': per_page,
        'pages': (len(sorted_history) + per_page - 1) // per_page
    })

# 生成价格预测
@app.route('/api/predictions', methods=['POST'])
def generate_prediction():
    data = request.json
    project_id = data.get('project_id')
    period = data.get('period', 30)  # 默认30天
    algorithm = data.get('algorithm', 'linear')  # 默认线性回归
    
    # 这里应该根据项目ID获取历史数据
    # 为简化示例，我们使用模拟数据生成预测
    history_data = generate_sample_price_data(60)  # 60天的历史数据
    
    # 基于历史数据生成预测
    if algorithm == 'linear':
        predictions = linear_regression_prediction(history_data, period)
    else:
        # 可以添加其他预测算法实现
        predictions = linear_regression_prediction(history_data, period)
    
    # 计算风险评估
    risk_assessment = assess_risk(predictions)
    
    return jsonify({
        'history_data': history_data,
        'predictions': predictions,
        'risk_assessment': risk_assessment,
        'accuracy': 87.2,  # 模拟预测准确率
        'algorithm': algorithm,
        'period': period
    })

# 处理风险数据文件
def process_risk_data(file_path):
    try:
        # 根据文件扩展名读取不同格式的文件
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # 这里可以添加数据处理和分析逻辑
        print(f"处理风险数据: {file_path}")
        print(f"数据形状: {df.shape}")
        print(f"数据列: {list(df.columns)}")
        
        # 示例：提取风险数据并更新风险列表
        # 实际应用中需要根据文件格式和内容进行适配
        
        return True
    except Exception as e:
        print(f"处理文件失败: {e}")
        return False

# 生成示例价格数据
def generate_sample_price_data(days):
    # 生成模拟价格数据（实际应用中应从数据库或文件获取）
    base_price = 120
    volatility = 0.02
    dates = [(datetime.now() - pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days-1, -1, -1)]
    prices = [base_price]
    
    for i in range(1, days):
        change = np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    return [{'date': date, 'price': round(price, 2)} for date, price in zip(dates, prices)]

# 线性回归价格预测
def linear_regression_prediction(history_data, period):
    # 提取历史价格数据
    prices = [item['price'] for item in history_data]
    dates = [i for i in range(len(prices))]
    
    # 训练线性回归模型
    X = np.array(dates).reshape(-1, 1)
    y = np.array(prices)
    
    model = LinearRegression()
    model.fit(X, y)
    
    # 生成预测日期和价格
    last_date = len(prices)
    prediction_dates = [(datetime.now() + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, period+1)]
    prediction_dates_numeric = np.array([i for i in range(last_date+1, last_date+period+1)]).reshape(-1, 1)
    
    # 预测价格
    predicted_prices = model.predict(prediction_dates_numeric)
    
    # 添加一些随机波动模拟不确定性
    volatility = 0.015
    predicted_prices = [price * (1 + np.random.normal(0, volatility)) for price in predicted_prices]
    
    return [{'date': date, 'price': round(price, 2)} for date, price in zip(prediction_dates, predicted_prices)]

# 风险评估
def assess_risk(predictions):
    # 基于预测价格计算风险评估
    prices = [item['price'] for item in predictions]
    price_changes = [(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices)-1)]
    
    avg_change = np.mean(price_changes)
    std_change = np.std(price_changes)
    
    # 计算风险等级
    risk_level = "低风险"
    if std_change > 0.03 or abs(avg_change) > 0.02:
        risk_level = "中风险"
    if std_change > 0.05 or abs(avg_change) > 0.03:
        risk_level = "高风险"
    
    return {
        'risk_level': risk_level,
        'volatility': round(std_change * 100, 2),  # 波动率百分比
        'avg_change': round(avg_change * 100, 2),  # 平均变化百分比
        'max_increase': round(max(price_changes) * 100, 2),
        'max_decrease': round(min(price_changes) * 100, 2)
    }

if __name__ == '__main__':
    app.run(debug=True)    