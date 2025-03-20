from flask import Flask, render_template, request, jsonify, session
import os
import json
import requests
import hashlib
import datetime
import random
import base64
import sqlite3
import uuid
from io import BytesIO
from PIL import Image
import pystray
from threading import Thread

app = Flask(__name__)
app.secret_key = 'fixed_secret_key_for_persistence'  # 使用固定密钥保证session持久化

# 定义数据库路径
DB_PATH = 'formula_history.db'

def init_db():
    """初始化数据库"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 创建历史记录表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        latex TEXT NOT NULL,
        conf REAL NOT NULL,
        timestamp TEXT NOT NULL,
        session_id TEXT NOT NULL
    )
    ''')
    
    # 创建设置表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS settings (
        session_id TEXT PRIMARY KEY,
        history_limit INTEGER NOT NULL DEFAULT 5
    )
    ''')
    
    conn.commit()
    conn.close()

# 应用启动时初始化数据库
init_db()

def get_session_id():
    """获取固定用户ID"""
    return 'fixed_user_id'  # 使用固定用户ID确保历史记录持久化

def get_history_limit(session_id):
    """获取历史记录限制数量"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT history_limit FROM settings WHERE session_id = ?', (session_id,))
    result = cursor.fetchone()
    
    if result:
        limit = result[0]
    else:
        # 默认值为5，并添加到数据库
        limit = 5
        cursor.execute('INSERT INTO settings (session_id, history_limit) VALUES (?, ?)', 
                     (session_id, limit))
        conn.commit()
    
    conn.close()
    return limit

def set_history_limit_db(session_id, limit):
    """设置历史记录限制数量"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('INSERT OR REPLACE INTO settings (session_id, history_limit) VALUES (?, ?)', 
                 (session_id, limit))
    
    # 如果当前历史记录数超过限制，删除最旧的记录
    cursor.execute('''
    DELETE FROM history 
    WHERE session_id = ? AND id NOT IN (
        SELECT id FROM history 
        WHERE session_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    )
    ''', (session_id, session_id, limit))
    
    conn.commit()
    conn.close()

def add_history_item(session_id, latex, conf):
    """添加历史记录项"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 获取当前时间
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 插入新记录
    cursor.execute('''
    INSERT INTO history (latex, conf, timestamp, session_id) 
    VALUES (?, ?, ?, ?)
    ''', (latex, conf, timestamp, session_id))
    
    # 获取历史记录限制
    limit = get_history_limit(session_id)
    
    # 删除超出限制的旧记录
    cursor.execute('''
    DELETE FROM history 
    WHERE session_id = ? AND id NOT IN (
        SELECT id FROM history 
        WHERE session_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    )
    ''', (session_id, session_id, limit))
    
    conn.commit()
    conn.close()

def get_history_items(session_id):
    """获取历史记录项"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 设置行工厂以便通过列名访问
    cursor = conn.cursor()
    
    # 获取当前历史限制
    limit = get_history_limit(session_id)
    
    # 查询历史记录
    cursor.execute('''
    SELECT id, latex, conf, timestamp 
    FROM history 
    WHERE session_id = ? 
    ORDER BY timestamp DESC 
    LIMIT ?
    ''', (session_id, limit))
    
    rows = cursor.fetchall()
    
    # 转换为字典列表
    history = []
    for row in rows:
        history.append({
            'id': row['id'],
            'latex': row['latex'],
            'conf': row['conf'],
            'timestamp': row['timestamp']
        })
    
    conn.close()
    return history, limit

def clear_history_items(session_id):
    """清空历史记录"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM history WHERE session_id = ?', (session_id,))
    
    conn.commit()
    conn.close()

def cleanup_temp_files():
    """清理临时文件夹中的所有临时图像文件"""
    try:
        for filename in os.listdir('.'):
            if filename.startswith('temp_') and filename.endswith('.png'):
                try:
                    os.remove(filename)
                except:
                    pass
    except Exception as e:
        print(f"清理临时文件时出错: {str(e)}")

# 应用启动时执行清理
cleanup_temp_files()

# 直接设置API配置
config = {
    "SIMPLETEX_APP_ID": "xxx",
    "SIMPLETEX_APP_SECRET": "xxx"
}

def random_str(randomlength=16):
    """Generate a random string"""
    chars = 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789'
    return ''.join(random.choice(chars) for _ in range(randomlength))

def get_req_data(req_data, appid, secret):
    """Generate API request data and header information"""
    header = {
        "timestamp": str(int(datetime.datetime.now().timestamp())),
        "random-str": random_str(16),
        "app-id": appid
    }
    pre_sign_string = "&".join(f"{k}={v}" for k, v in sorted({**header, **req_data}.items()))
    pre_sign_string += "&secret=" + secret
    header["sign"] = hashlib.md5(pre_sign_string.encode()).hexdigest()
    return header, req_data

@app.route('/')
def index():
    # 确保用户有一个session_id
    get_session_id()
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize_formula():
    if 'file' not in request.files and 'image_data' not in request.form:
        return jsonify({'status': False, 'msg': '没有提供图像'})
    
    try:
        # 生成唯一的临时文件名，避免冲突
        temp_file = f"temp_{uuid.uuid4().hex}.png"
        
        # Handle file upload or base64 image data
        if 'file' in request.files:
            file = request.files['file']
            file.save(temp_file)
        else:
            # Handle base64 image data (from clipboard)
            image_data = request.form['image_data']
            image_data = image_data.split(',')[1] if ',' in image_data else image_data
            with open(temp_file, 'wb') as f:
                f.write(base64.b64decode(image_data))
        
        # Check if config is complete
        if not config['SIMPLETEX_APP_ID'] or not config['SIMPLETEX_APP_SECRET']:
            return jsonify({'status': False, 'msg': 'API配置信息不完整'})
        
        # API call logic
        data = {}
        header, data = get_req_data(data, config['SIMPLETEX_APP_ID'], config['SIMPLETEX_APP_SECRET'])
        
        # 使用with语句确保文件在使用后正确关闭
        with open(temp_file, 'rb') as f:
            img_file = {"file": f}
            res = requests.post("https://server.simpletex.cn/api/latex_ocr", 
                              files=img_file, data=data, headers=header)
        
        # 确保文件已关闭后再尝试删除
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            print(f"无法删除临时文件: {str(e)}")
            # 继续处理，不让临时文件错误影响主流程
            
        result = json.loads(res.text)
        
        # 如果识别成功，添加到历史记录
        if result.get('status', False) and 'res' in result:
            # 获取当前会话ID
            session_id = get_session_id()
            
            # 添加到历史记录
            add_history_item(
                session_id=session_id,
                latex=result['res']['latex'],
                conf=result['res']['conf']
            )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'status': False, 'msg': f'错误: {str(e)}'})

@app.route('/get_history', methods=['GET'])
def get_history():
    """获取历史记录"""
    session_id = get_session_id()
    history, limit = get_history_items(session_id)
    
    return jsonify({
        'status': True,
        'history': history,
        'history_limit': limit
    })

@app.route('/set_history_limit', methods=['POST'])
def set_history_limit():
    """设置历史记录限制"""
    try:
        limit = int(request.form.get('limit', 5))
        # 限制范围在1-20之间
        limit = max(1, min(20, limit))
        
        session_id = get_session_id()
        set_history_limit_db(session_id, limit)
        
        return jsonify({'status': True, 'msg': f'历史记录限制已设置为 {limit}'})
    except Exception as e:
        return jsonify({'status': False, 'msg': f'设置失败: {str(e)}'})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """清空历史记录"""
    session_id = get_session_id()
    clear_history_items(session_id)
    return jsonify({'status': True, 'msg': '历史记录已清空'})

def create_tray_icon():
    # 创建系统托盘图标
    try:
        # 尝试从打包后的资源中加载图标
        import pkg_resources
        icon_data = pkg_resources.resource_string(__name__, 'static/favicon.ico')
        image = Image.open(BytesIO(icon_data))
    except:
        # 如果失败，尝试从文件系统加载
        image = Image.open("static/favicon.ico")
    
    menu = (
        pystray.MenuItem('跳转', lambda: webbrowser.open('http://127.0.0.1:5000')),
        pystray.MenuItem('退出', lambda: os._exit(0)),
    )
    
    icon = pystray.Icon("mathweb", image, "MathWeb", menu)
    icon.run()

if __name__ == '__main__':
    import webbrowser
    webbrowser.open('http://127.0.0.1:5000')
    
    # 启动系统托盘图标
    tray_thread = Thread(target=create_tray_icon)
    tray_thread.daemon = True
    tray_thread.start()
    
    # 启动Flask应用
    app.run(debug=True, use_reloader=False)
