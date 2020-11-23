# -*- coding: utf-8 -*-
import json
import urllib
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler

from OutputFusion.outputFusion import format_result


class ServerHTTP(BaseHTTPRequestHandler):
    def do_GET(self):

        path = self.path
        # 拆分url(也可根据拆分的url获取Get提交才数据),可以将不同的path和参数加载不同的html页面，或调用不同的方法返回不同的数据，来实现简单的网站或接口
        query = urllib.splitquery(path)
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("test", "This is test!")
        self.end_headers()
        buf = '''<!DOCTYPE HTML>
        <html>
        <head><title>Get page</title></head>
        <body>
        <form action="post_page" method="post">
            userId: <input type="text" name="userId" /><br />
            itemId: <input type="text" name="itemId" /><br />
            listNumber: <input type="text" name="listNumber" /><br />
            <input type="submit" value="recommend" />
        </form>
        </body>
        </html>'''
        self.wfile.write(buf)

    def do_POST(self):
        path = self.path
        datas = self.rfile.read(int(self.headers['content-length']))

        result = {}
        for i in datas.split("&"):
            (key, value) = i.split("=")
            result[key] = value;
        results = format_result(int(result["userId"]),int(result["listNumber"]))

        datas = {}
        datas["status"] = 200
        datas["message"] = "OK"
        datas["data"] = results
        datas = json.dumps(datas)

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("test", "This is test!")
        self.end_headers()
        buf = '''<!DOCTYPE HTML>
        <html>
            <head><title>Post page</title></head>
            <body>Post Data:%s <br />Path:%s</body>
        </html>'''%(datas, self.path)
        self.wfile.write(datas)

def start_server(port):
    http_server = HTTPServer(('', int(port)), ServerHTTP)
    http_server.serve_forever()

if __name__ == "__main__":
    start_server(8000)