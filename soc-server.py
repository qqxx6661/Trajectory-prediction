# receive

import socket
import hashlib
import os
import struct

HOST = 'localhost'
PORT = 1307
BUFFER_SIZE = 1024
HEAD_STRUCT = '128sIq32s'
info_size = struct.calcsize(HEAD_STRUCT)


def cal_md5(file_path):
    with open(file_path, 'rb') as fr:
        md5 = hashlib.md5()
        md5.update(fr.read())
        md5 = md5.hexdigest()
        return md5


def unpack_file_info(file_info):
    file_name, file_name_len, file_size, md5 = struct.unpack(HEAD_STRUCT, file_info)
    file_name = file_name[:file_name_len]
    return file_name, file_size, md5


def recv_file():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (HOST, PORT)
        sock.bind(server_address)
        sock.listen(1)
        client_socket, client_address = sock.accept()
        print "Connected %s successfully" % str(client_address)

        file_info_package = client_socket.recv(info_size)
        file_name, file_size, md5_recv = unpack_file_info(file_info_package)

        recved_size = 0
        with open(file_name, 'wb') as fw:
            while recved_size < file_size:
                remained_size = file_size - recved_size
                recv_size = BUFFER_SIZE if remained_size > BUFFER_SIZE else remained_size
                recv_file = client_socket.recv(recv_size)
                recved_size += recv_size
                fw.write(recv_file)
        md5 = cal_md5(file_name)
        if md5 != md5_recv:
            print 'MD5 compared fail!'
        else:
            # os.system('python /home/zhendongyang/PycharmProjects/EaML/CLOUD_ML.py')
            print 'Received successfully'
    except socket.errno, e:
        print "Socket error: %s" % str(e)
    finally:
        sock.close()

if __name__ == '__main__':
    recv_file()