# -*- coding: utf-8 -*-
from OutputFusion.outputFusion import format_result

# -*- coding: UTF-8 -*-
import os
import sys
import json
from gevent import monkey
from flask import Flask, request
from gevent.pywsgi import WSGIServer
# from OutputFusion.outputFusion import format_result_with_image

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

monkey.patch_all()
app = Flask(__name__)


# paddleocr文字识别
@app.route('/interfaceImage', methods=['POST', 'GET'])
def interfaceImage():
    user_id = request.form['userId']
    list_number = request.form['listNumber']
    results = format_result(str(user_id), int(list_number))
    return json.dumps(results, ensure_ascii=False)


if __name__ == "__main__":
    WSGIServer(('0.0.0.0', 8000), app).serve_forever()


# class ServerHTTP(BaseHTTPRequestHandler):
#     def do_GET(self):
#         logging.warning(u"运行日志：接口GET")
#         path = self.path
#         # 拆分url(也可根据拆分的url获取Get提交才数据),可以将不同的path和参数加载不同的html页面，或调用不同的方法返回不同的数据，来实现简单的网站或接口
#         # query = urllib.splitquery(path)
#         self.send_response(200)
#         self.send_header("Content-type", "text/html")
#         self.send_header("test", "This is test!")
#         self.end_headers()
#         buf = '''<!DOCTYPE HTML>
#         <html>
#         <head><title>Get page</title></head>
#         <body>
#         <form action="post_page" method="post">
#             userId: <input type="text" name="userId" /><br />
#             itemId: <input type="text" name="itemId" /><br />
#             listNumber: <input type="text" name="listNumber" /><br />
#             <input type="submit" value="recommend" />
#         </form>
#         </body>
#         </html>'''
#         self.wfile.write(str.encode(buf))
#
#     def do_POST(self):
#         logging.warning(u"运行日志：接口POST")
#         path = self.path
#         datas = self.rfile.read(int(self.headers['content-length']))
#         print(datas.decode())
#
#         temp = {}
#         for i in datas.split(str.encode("&")):
#             print(i)
#             (key, value) = i.split(str.encode("="))
#             temp[key] = value;
#         result = {}
#         for key, value in temp.items():
#             result[bytes.decode(key)] = bytes.decode(value)
#         #results = format_result(int(result["userId"]),int(result["listNumber"]))
#         results = format_result(str(result["userId"]),int(result["listNumber"]))
#
#         datas = {}
#         datas["status"] = 200
#         datas["message"] = "OK"
#         datas["data"] = results
#         datas = json.dumps(datas, ensure_ascii=False)
#
#         self.send_response(200)
#         self.send_header("Content-type", "text/html")
#         self.send_header("test", "This is test!")
#         self.end_headers()
#         buf = '''<!DOCTYPE HTML>
#         <html>
#             <head><meta charset="utf-8">
#             <title>Post page</title>
#             </head>
#             <body>Post Data:%s <br />Path:%s</body>
#         </html>'''%(datas, self.path)
#         self.wfile.write(str.encode(buf))
#
# def start_server(port):
#     http_server = HTTPServer(('', int(port)), ServerHTTP)
#     http_server.serve_forever()
#
# if __name__ == "__main__":
#     start_server(8000)