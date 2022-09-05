# -*- coding: utf-8 -*-
# ----------------------------------------------------#
#   工具箱
# ----------------------------------------------------#
import torch
def writelog(file, log_info):
    '''
    写日志
    :param file: 日志路径
    :param log_info: 日志信息
    :return:
    '''
    with open(file, 'a') as f:
        f.write(log_info + '\n')
    print(log_info)
def save_checkpoint_state(path, epoch, model, optimizer, scheduler,scaler):
    '''
    存模型
    :param path: 保存路径
    :param epoch: 训练第几轮
    :param model: 模型对象
    :param optimizer: 梯度优化器对象
    :param scheduler: 学习率优化器对象
    :param scaler: 标准化器
    :return: void
    '''

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler': scaler
    }
    torch.save(checkpoint, path)
def save_checkpoint(path,model,k_features_scaler,tricks_features_scaler):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'k_features_scaler': k_features_scaler,
        'tricks_features_scaler': tricks_features_scaler,
    }
    torch.save(checkpoint, path)
def get_checkpoint_state(model_path, model, optimizer, scheduler):
    '''
    恢复上次的训练状态
    :param model_path:
    :param model:
    :param optimizer:
    :param scheduler:
    :return:
    '''
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    scaler = checkpoint['scaler']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return model, epoch, optimizer, scheduler,scaler
def norm(model,loss):
    '''l1正则'''
    l1_reg = torch.tensor(0., requires_grad=True)
    for name, param in model.named_parameters():
        if 'weight' in name:
            l1_reg = l1_reg + torch.norm(param, 1)
    loss = loss + 10e-4 * l1_reg
    return loss
# [1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]
# [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1],
# [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0]