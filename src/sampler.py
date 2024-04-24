import copy
import torch
import numpy as np
import torch.nn as nn


class SGLD(object):
    '''Class for Stochastic Gradient Langevin Dynamics'''
    def __init__(self, args):

        self.args = args
        
        self.sample_num = args['sample_num']
        self.select_p = args['select_p']
        if self.select_p < -1e-5 or self.select_p-1 > 1e-5:
            raise ValueError("SGLD probability should in [0, 1].")
        self.rotation_range = [-torch.pi, torch.pi]
        self.translation_range = [-args['max_translation'], args['max_translation']]
        self.translation_std_rec = args['translation_std_rec']
        self.translation_std_lig = args['translation_std_lig']
        self.sgld_lr = args['sgld_lr']
        self.sgld_R_var = args['sgld_R_var']
        self.sgld_R_clip = args['sgld_R_clip']
        self.sgld_T_var = args['sgld_T_var']
        self.sgld_T_clip = args['sgld_T_clip']
        self.iters = args['sample_iters']

    def sample(self, model, R, T, 
               pos_lig, pos_rec, gaussians_params, mdn_mask, 
               if_train=False, early_stop=False):

        if if_train:
            model.train()
        
        noise_R = torch.rand_like(R[0])
        noise_T = torch.rand_like(T[0])
        
        for i in range(self.sample_num):
            R_route = []
            T_route = []
            if early_stop:
                wait_step = 0
                bestRi = None
                bestTi = None
                bestenergy = 1e9
            
            for _ in range(self.iters):

                noise_R.normal_(0, self.sgld_R_var)
                noise_T.normal_(0, self.sgld_T_var)

                R.data[i].add_(noise_R)
                T.data[i].add_(noise_T)

                while torch.any(R.data[i] > self.rotation_range[1]):
                    R.data[i][torch.where(R.data[i]>self.rotation_range[1])] -= 2*torch.pi
                while torch.any(R.data[i] < self.rotation_range[0]):
                    R.data[i][torch.where(R.data[i]<self.rotation_range[0])] += 2*torch.pi
                T.data[i].clamp_(self.translation_range[0], self.translation_range[1])
                
                Ri, Ti = R[i], T[i]
                energyall = model(pos_lig, pos_rec, gaussians_params, Ri, Ti, mdn_mask=mdn_mask)
                energyall = energyall.sum()                
                if early_stop:
                    if energyall < bestenergy:
                        bestenergy = energyall.item()
                        bestRi = Ri.detach().clone()
                        bestTi = Ti.detach().clone()
                        wait_step = 0
                    else:
                        wait_step += 1
                    if wait_step > 20:
                        break

                R_grads, T_grads = torch.autograd.grad(energyall, [Ri, Ti], retain_graph=True)
                R_grads.data.clamp_(-self.sgld_R_clip, self.sgld_R_clip)
                T_grads.data.clamp_(-self.sgld_T_clip, self.sgld_T_clip)

                R_grads_norm = R_grads.norm()                
                if R_grads_norm > 2:
                    R_lr_time = 10
                elif R_grads_norm > 0.5:
                    R_lr_time = 15
                else:
                    R_lr_time = 20

                T_grads_norm = T_grads.norm()
                if T_grads_norm > 2:
                    T_lr_time = 200
                elif T_grads_norm > 0.5:
                    T_lr_time = 300
                else:
                    T_lr_time = 400

                R.data[i].add_(R_grads, alpha=-self.sgld_lr*R_lr_time)
                T.data[i].add_(T_grads, alpha=-self.sgld_lr*T_lr_time)

            while torch.any(R.data[i] > self.rotation_range[1]):
                R.data[i][torch.where(R.data[i]>self.rotation_range[1])] -= 2*torch.pi
            while torch.any(R.data[i] < self.rotation_range[0]):
                R.data[i][torch.where(R.data[i]<self.rotation_range[0])] += 2*torch.pi
        
            T.data[i].clamp_(self.translation_range[0], self.translation_range[1])
            Ri = R[i]
            Ti = T[i]
            energyall = model(pos_lig, pos_rec, gaussians_params, R=Ri, t=Ti, mdn_mask=mdn_mask)
            energyall = energyall.sum()
            if early_stop:
                if energyall < bestenergy:
                    pass
                else:
                    R.data[i] = bestRi
                    T.data[i] = bestTi

        return R, T


    def random_sample(self, receptor_center, ligand_center=None):
        '''
        rotation_range: (-pi, pi) by default
        translation_range: (a, b)
        batchsize: n

        TODO:
        uniform distribution now, may change to others?
        '''
        rotation_all = []
        translation_all = []
        for i in range(self.sample_num):
            rotation = torch.rand(len(receptor_center),3,requires_grad=True)*(self.rotation_range[1]-self.rotation_range[0])+self.rotation_range[0]
            if (self.select_p < 1.) and (ligand_center is not None):
                if torch.rand(1).item() < self.select_p:
                    translation = torch.randn(len(receptor_center),3,requires_grad=True)*self.translation_std_rec + receptor_center
                else:
                    translation = torch.randn(len(receptor_center),3,requires_grad=True)*self.translation_std_lig + ligand_center
            else:
                translation = torch.randn(len(receptor_center),3,requires_grad=True)*self.translation_std_rec + receptor_center
            translation.data.clamp_(self.translation_range[0], self.translation_range[1])
            rotation_all.append(rotation)
            translation_all.append(translation)
        rotation_all = torch.stack(rotation_all)
        translation_all = torch.stack(translation_all)

        return rotation_all.to(self.args['device']), translation_all.to(self.args['device'])
    
