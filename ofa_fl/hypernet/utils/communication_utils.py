import pickle
import socket
import time
from tqdm import tqdm
import math

def recv(soc, buffer_size=1024, recv_timeout=10):
    # get message length
    try:
        soc.settimeout(recv_timeout)
        soc.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        soc.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)
        msg = soc.recv(buffer_size)
        msg = pickle.loads(msg)
        if msg['subject'] == 'header':
            data_len = msg['data']
            soc.sendall(pickle.dumps({"subject": "header", "data": "ready"}))
        else:
            raise Exception('Does not receive message header.')
            return None, 0
    except socket.timeout:
        print(f"A socket.timeout exception occurred after {recv_timeout} seconds. There may be an error or the model may be trained successfully.")
        return None, 0
    except BaseException as e:
        print(f"An error occurred while receiving header {e}.")
        return None, 0

    received_data = bytearray(data_len)
    view = memoryview(received_data)
    total_received = 0
    while total_received < data_len:
        try:
            remaining = data_len - total_received
            chunk_size = min(buffer_size, remaining)
            msg = soc.recv_into(view[total_received:total_received + chunk_size])
            if not msg:
                break
            total_received += msg
        except socket.timeout:
            print(
                f"A socket.timeout exception occurred after {recv_timeout} seconds. There may be an error or the model may be trained successfully.")
            return None, 0
        except BaseException as e:
            print(f"An error occurred while receiving data {e}.")
            return None, 0
    received_data = bytes(received_data)
    try:
        received_data = pickle.loads(received_data)
    except BaseException as e:
        print(f"Error Decoding the Client's Data: {e}.")
        return None, 0

    return received_data, 1


def send(soc, msg, buffer_size=1024):
    # msg: data bytes
    soc.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)
    soc.sendall(pickle.dumps({"subject": "header", "data": len(msg)}))
    soc.recv(buffer_size)
    soc.sendall(msg)
