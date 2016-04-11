'''from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket

class SimpleEcho(WebSocket):

    def handleMessage(self):
        # echo message back to client
        print self.data

    def handleConnected(self):
        print self.address, 'connected'

    def handleClose(self):
        print self.address, 'closed'

server = SimpleWebSocketServer('localhost', 9999, SimpleEcho)
server.serveforever()
'''
import BaseHTTPServer, SimpleHTTPServer, SocketServer
import ssl

class MyTCPHandler(SocketServer.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def handle(self):
        # self.request is the TCP socket connected to the client
        self.data = self.request.recv(1024)
        print "{} wrote:".format(self.client_address[0])
        print len(self.data)
        print self.data

httpd = BaseHTTPServer.HTTPServer(('localhost', 9999), MyTCPHandler)
httpd.socket = ssl.wrap_socket (httpd.socket, certfile='cert.pem', server_side=True)
httpd.serve_forever()