import sys
import time
import struct
import statistics

from task.helper import get_data
from util.util import TcpClient, timestamp

def send_request(client, task_name, data):
    timestamp('client', 'before_request_%s' % task_name)

    # Serialize data
    task_name_b = task_name.encode()
    task_name_length = len(task_name_b)
    task_name_length_b = struct.pack('I', task_name_length)

    if data is not None:
        data_b = data.numpy().tobytes()
        length = len(data_b)
    else:
        data_b = None
        length = 0
    length_b = struct.pack('I', length)
    # timestamp('client', 'after_inference_serialization')

    # Send Data
    client.send(task_name_length_b)
    client.send(task_name_b)
    client.send(length_b)
    if data_b is not None:
        client.send(data_b)
    timestamp('client', 'after_request_%s' % task_name)

def recv_response(client):
    reply_b = client.recv(4)
    reply = reply_b.decode()
    timestamp('client', 'after_reply')

def close_connection(client):
    model_name_length = 0
    model_name_length_b = struct.pack('I', model_name_length)
    client.send(model_name_length_b)
    timestamp('client', 'close_connection')

def main():
    timestamp('frontend', 'start')
    
    # Load model list
    model_list_file_name = sys.argv[1]
    task_list = []
    with open(model_list_file_name) as f:
        for line in f.readlines():
            task_list.append(line.strip())
    
    for i in range (len(task_list)):
        model_name = task_list[i].split()[0]
        batch_size = int(task_list[i].split()[1])
        task_name_train = '%s_training' % model_name
        data = get_data(model_name, batch_size)
        client_train = TcpClient('localhost', 12346)
        send_request(client_train, task_name_train, data)
        recv_response(client_train)
        close_connection(client_train)
        timestamp('**********', '**********')


    # for i in range (40):
    #     model_name = task_list[i].split()[0]
    #     batch_size = int(task_list[i].split()[1])
    #     task_name_train = '%s_training' % model_name
    #     data = get_data(model_name, batch_size)
    #     client_train = TcpClient('localhost', 12346)
    #     send_request(client_train, task_name_train, data)
    #     recv_response(client_train)
    #     close_connection(client_train)
    #     timestamp('**********', '**********')
    #     time.sleep(2)

    # model_name1 = sys.argv[1]
    # model_name2 = sys.argv[2]
    # batch_size = int(sys.argv[3])

    # task_1_name_train = '%s_training' % model_name1
    # task_2_name_train = '%s_training' % model_name2

    # # Load image
    # data1 = get_data(model_name1, batch_size)
    # data2 = get_data(model_name2, batch_size)

    # latency_list = []
    # for _ in range(4):
    #     # Send training request
    #     client_train_1 = TcpClient('localhost', 12346)
    #     send_request(client_train_1, task_1_name_train, data1)
    #     time.sleep(4)

    #     # Connect
    #     client_train_2 = TcpClient('localhost', 12346)
    #     timestamp('client', 'after_inference_connect')
    #     time_1 = time.time()

    #     # Send inference request
    #     send_request(client_train_2, task_2_name_train, data2)

    #     # Recv inference reply
    #     recv_response(client_train_2)
    #     time_2 = time.time()
    #     latency = (time_2 - time_1) * 1000
    #     latency_list.append(latency)

    #     time.sleep(1)
    #     recv_response(client_train_1)
    #     close_connection(client_train_2)
    #     close_connection(client_train_1)
    #     time.sleep(1)
    #     timestamp('**********', '**********')

    # print()
    # print()
    # print()
    # stable_latency_list = latency_list[1:]
    # print (stable_latency_list)
    # print ('Latency: %f ms (stdev: %f)' % (statistics.mean(stable_latency_list), 
    #                                        statistics.stdev(stable_latency_list)))

if __name__ == '__main__':
    main()
