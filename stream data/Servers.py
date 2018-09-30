import socket
import pickle
import struct
import logging

from threading import Thread


logging.basicConfig(level=logging.INFO)
log = logging.getLogger('Server')


class Server(object):
    def __init__(self, server_address,queue):
        self.connection(server_address)
        self.queue = queue
        self.t = Thread(target=self.Receive_thread)
        self.t.start()

    def connection(self, server_address):
        # Create a TCP/IP socket
        self.sock = socket.socket()
        log.info('starting up on %s port %s' % server_address)
        self.sock.bind(server_address)
        self.sock.listen(10)

        # Connect the socket
        self.connection, self.client_address = self.sock.accept()
        log.info('connection from' + str(self.client_address))

    def getClient(self):
        return self.connection, self.client_address



    def recvall(self, sock, n):
        # Helper function to recv n bytes or return None if EOF is hit
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def Receive_thread(self):
        while True:
            log.debug("waiting data from client...")
            raw_msglen = self.recvall(self.connection, 4)
         
            if raw_msglen:
                msglen = int(raw_msglen.decode())
                # Read the message data
                data = self.recvall(self.connection, msglen)
                log.debug("Receive complete")
                data_arr = data.decode().split(",")
                self.queue.put(data_arr)

    def close(self):
        self.SendtoClient('0')
        self.sock.close()
        log.info('closing socket')