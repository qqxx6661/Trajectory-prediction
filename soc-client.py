import socket
import os
import hashlib
import struct
import time

HOST = '54.149.245.21'
PORT = 8000
BUFFER_SIZE = 1024
HEAD_STRUCT = '128sIq32s'


def cal_md5(file_path):
    with open(file_path, 'rb') as fr:
        md5 = hashlib.md5()
        md5.update(fr.read())
        md5 = md5.hexdigest()
        return md5


def get_file_info(file_path):
    file_name = os.path.basename(file_path)
    file_name_len = len(file_name)
    file_size = os.path.getsize(file_path)
    md5 = cal_md5(file_path)
    return file_name, file_name_len, file_size, md5


def send_file(file_path):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (HOST, PORT)

    file_name, file_name_len, file_size, md5 = get_file_info(file_path)
    file_head = struct.pack(HEAD_STRUCT, file_name, file_name_len, file_size, md5)

    try:
        print "Start connect"
        start_time = time.time()
        sock.connect(server_address)
        sock.send(file_head)
        sent_size = 0

        with open(file_path) as fr:
            while sent_size < file_size:
                remained_size = file_size - sent_size
                send_size = BUFFER_SIZE if remained_size > BUFFER_SIZE else remained_size
                send_file = fr.read(send_size)
                sent_size += send_size
                sock.send(send_file)
    except socket.errno, e:
        print "Socket error: %s" % str(e)
    finally:
        sock.close()
        print "Closing connect"
        end_time = time.time()
        print end_time - start_time

if __name__ == '__main__':
    file_path = raw_input('Please input file path:')
    if not file_path:
        file_path = 'EDGE/video12.9-3FPS/2017-12-09 14-08-30_2.avi'
    send_file(file_path)