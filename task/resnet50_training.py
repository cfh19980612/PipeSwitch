import time

import torch
import torch.nn as nn

import statistics

import task.resnet50 as resnet50
import task.common as util

TASK_NAME = 'resnet50_training'

def import_data_loader():
    return resnet50.import_data

def import_model():
    model = resnet50.import_model()
    model.train()
    return model

def import_func():
    def train(model, data_loader):
        training_time_list = []
        # Prepare data
        batch_size = 32
        images, target = data_loader(batch_size)

        # Prepare training
        lr = 0.1
        momentum = 0.9
        weight_decay = 1e-4
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

        loss = None
        for i in range(10):
            start_time = time.time()
            # Data to GPU
            images_cuda = images.cuda(non_blocking=True)
            target_cuda = target.cuda(non_blocking=True)

            # compute output
            output = model(images_cuda)
            loss = criterion(output, target_cuda)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_time = time.time()
            print ('Training', i, time.time(), loss.item())
            training_time_list.append((end_time - start_time)*1000)
            del images_cuda
            del target_cuda
        print ('Latency: %f ms (stdev: %f)' % (statistics.mean(training_time_list), 
                                           statistics.stdev(training_time_list)))
        return loss.item()
    return train

def import_task():
    model = import_model()
    func = import_func()
    group_list = resnet50.partition_model(model)
    shape_list = [util.group_to_shape(group) for group in group_list]
    return model, func, shape_list

def import_parameters():
    model = import_model()
    group_list = resnet50.partition_model(model)
    batch_list = [util.group_to_batch(group) for group in group_list]
    return batch_list