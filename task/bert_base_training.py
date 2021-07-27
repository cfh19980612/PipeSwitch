import time

import torch
import torch.nn as nn

import task.bert_base as bert_base
import task.common as util
import statistics

TASK_NAME = 'bert_base_training'

def import_data_loader():
    return bert_base.import_data

def import_model():
    model = bert_base.import_model()
    model.train()
    return model

def import_func():
    def train(model, data_loader):
        # Prepare data
        batch_size = 12
        data, target = data_loader(batch_size)

        # Prepare training
        criterion = nn.MSELoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        loss = None
        training_time_list = []
        for i in range(10):
            # Data to GPU
            data_cuda = data.view(2, -1, 251).cuda()
            target_0_cuda = target[0].cuda()
            target_1_cuda = target[1].cuda()

            # compute output
            start_time = time.time()
            output = model(data_cuda[0], token_type_ids=data_cuda[1])
            loss1 = criterion(output[0], target_0_cuda)
            loss2 = criterion(output[1], target_1_cuda)
            loss = loss1 + loss2

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_time = time.time()
            print ('Training', i, time.time(), loss.item())
            del data_cuda
            del target_0_cuda
            del target_1_cuda
            training_time_list.append((end_time - start_time)*1000)
        print ('%s, Latency: %f ms (stdev: %f)' % (TASK_NAME, statistics.mean(training_time_list), 
                                           statistics.stdev(training_time_list)))
        return loss
    
    return train

def import_task():
    model = import_model()
    func = import_func()
    group_list = bert_base.partition_model(model)
    shape_list = [util.group_to_shape(group) for group in group_list]
    return model, func, shape_list

def import_parameters():
    model = import_model()
    group_list = bert_base.partition_model(model)
    batch_list = [util.group_to_batch(group) for group in group_list]
    return batch_list