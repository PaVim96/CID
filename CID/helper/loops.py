from __future__ import print_function, division

import sys
import time
import torch

import torch.nn as nn

from CID.helper.util import AverageMeter, accuracy
from CID.helper.util import cluster




def train_no_int(epoch, train_loader, module_list, criterion_list, optimizer, opt): 
    # set modules as train()
    for module in module_list:
        module.train()
    
    # set teacher as eval()
    module_list[-1].eval()

    #P.V: 
    #Criterion_list[0] => Cross Entropy Loss
    #Criterion_list[1] => KL Divergence 
    #Criterion_list[2] => Normalized MSE 
    #Criterion_list[3] => Cosine Loss
     
    
    #criterion_kl = criterion_list[1]
    criterion_cls = criterion_list[0]
    criterion_mse = criterion_list[1]
    criterion_sp = criterion_list[2]
    
    softmax = nn.Softmax(dim=1).cuda()


    model_s = module_list[0]
    model_t = module_list[2]

    #the prediction layer
    #not needed because we dont need to modify the prediction layer
    #model_s_fc_new = module_list[1]

    #the mapping layer from student feature space of hint layer to teacher hint layer
    fea_reg = module_list[1]


    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    class_num = model_s.fc.weight.shape[0]


    for idx, data in enumerate(train_loader): 
        input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
  
        # ===================forward=====================
        preact = False

        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        with torch.no_grad():
            feat_t, _ = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]


        f_s = fea_reg(feat_s[opt.hint_layer])
        f_t = feat_t[opt.hint_layer]

        loss_sample = criterion_mse(f_s, f_t)
        #list of sample representations of both student and teacher model
        list_s, list_t = cluster(feat_s[opt.hint_layer], f_t, target, class_num)
            
        involve_class = 0
            
        loss_class=0.0
            
        #class representation loss
        for k in range( len(list_s) ):
            
            cur_len = len( list_s[k] )
            
            if cur_len>=2:
                cur_f_s = torch.stack(list_s[k])
                cur_f_t = torch.stack(list_t[k])
                    
                loss_class+= criterion_sp(cur_f_s, cur_f_t) 
                    
                involve_class += 1
                    
                    
        if involve_class==0:
            loss_class = 0.0
        else:
            loss_class = loss_class/involve_class   


        softened_logit_s = softmax(logit_s / opt.net_T)
        loss_cls = criterion_cls(softened_logit_s, target)
        #loss = P(Y|X) + Sample Representation Loss + Class Representation Loss
        loss = opt.aa * loss_cls +  opt.bb * loss_sample + opt.cc * loss_class
        

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()
        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    
    #P.V: Since this does not use context, I guess this does not contain the interventional loss?
    return top1.avg, losses.avg

        

def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    
    # set modules as train()
    for module in module_list:
        module.train()
    
    # set teacher as eval()
    module_list[-1].eval()

    #P.V: 
    #Criterion_list[0] => Cross Entropy Loss
    #Criterion_list[1] => KL Divergence 
    #Criterion_list[2] => Normalized MSE 
    #Criterion_list[3] => Cosine Loss 
    #criterion_cls = criterion_list[0]
    criterion_kl = criterion_list[1]
    criterion_mse = criterion_list[2]
    criterion_sp = criterion_list[3]
    
    softmax = nn.Softmax(dim=1).cuda()


    model_s = module_list[0]
    model_t = module_list[-1]
    

    #TODO: this is not needed when not doing interventions
    try:
        context_new = torch.zeros( model_s.fc.weight.shape, dtype=torch.float32).cuda()   
        current_num = torch.zeros(model_s.fc.weight.shape[0], dtype=torch.float32).cuda()
        class_num = model_s.fc.weight.shape[0]
    except:
        try:
            context_new = torch.zeros( model_s.linear.weight.shape, dtype=torch.float32).cuda()   
            current_num = torch.zeros(model_s.linear.weight.shape[0], dtype=torch.float32).cuda()
            class_num = model_s.linear.weight.shape[0]
        except:
            context_new = torch.zeros( model_s.classifier.weight.shape, dtype=torch.float32).cuda()   
            current_num = torch.zeros(model_s.classifier.weight.shape[0], dtype=torch.float32).cuda()
            class_num = model_s.classifier.weight.shape[0]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    
    for idx, data in enumerate(train_loader):
  
        input, target, index = data
        
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
  
        # ===================forward=====================
        preact = False

        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]
            

        #if init epochs are reached, context is constructed
        if epoch==opt.init_epochs:   
            
            fea_s  = feat_s[opt.hint_layer].detach()
            
            soft_t = softmax(logit_t/opt.net_T)
            

            #not needed when no interventions are used
            for i in range( len(target) ):
                #soft_t[i][target[i]] => P(c_i | x_j) 
                #current_num => sum of all P(c_i| x_j)
                #I think context_new[target[i]] => c_i? 
                context_new[target[i]] = context_new[target[i]]*( current_num[target[i]]/(current_num[target[i]]+soft_t[i][target[i]]) )+ fea_s[i]*(soft_t[i][target[i]]/(current_num[target[i]]+soft_t[i][target[i]]))
                current_num[target[i]]+= soft_t[i][target[i]]
        
        loss_kl = criterion_kl(logit_s, logit_t)
        
        fea_reg = module_list[2]
        #fea_reg maps feature space of student model to feature space of teacher model
        #based on FitNet: https://arxiv.org/pdf/1412.6550.pdf
        #f_s and f_t are the features of the last layer => sample representations 
        f_s = fea_reg(feat_s[opt.hint_layer])
        f_t = feat_t[opt.hint_layer]
            
        #Sample Representation Loss
        loss_sample = criterion_mse(f_s, f_t)


        #list of sample representations of both student and teacher model
        list_s, list_t = cluster(feat_s[opt.hint_layer], f_t, target, class_num)
            
        involve_class = 0
            
        loss_class=0.0
            
        #class representation loss
        for k in range( len(list_s) ):
            
            cur_len = len( list_s[k] )
            
            if cur_len>=2:
                cur_f_s = torch.stack(list_s[k])
                cur_f_t = torch.stack(list_t[k])
                    
                loss_class+= criterion_sp(cur_f_s, cur_f_t) 
                    
                involve_class += 1
                    
                    
        if involve_class==0:
            loss_class = 0.0
        else:
            loss_class = loss_class/involve_class   
        
        #P.V: 
        #interventional loss should be KL + P(Y|f_s(x) & sum P(c_i) a_i^s * class representation)
        #P(Y|f_s(x)) => student prediction => f_s ? 
        #so disabling interventional loss should be same loss without kl part
        #question: do we disable KL or not? because its not inherently part of the intervention
        #just forces the context to be similar 
        loss =  opt.aa*loss_kl + opt.bb * loss_sample + opt.cc * loss_class 

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    
    #P.V: Since this does not use context, I guess this does not contain the interventional loss?
    return top1.avg, losses.avg, context_new



def train_distill_context(epoch, train_loader, module_list, criterion_list, optimizer, opt, context):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    
    # set teacher as eval()
    module_list[-1].eval()


    context_new = torch.zeros(context.shape, dtype=torch.float32).cuda()
   
    current_num = torch.zeros(context.shape[0], dtype=torch.float32).cuda()


    #Criterion_list[0] => Cross Entropy Loss
    #Criterion_list[1] => KL Divergence 
    #Criterion_list[2] => Normalized MSE 
    #Criterion_list[3] => Cosine Loss
    criterion_cls = criterion_list[0]
    criterion_kl = criterion_list[1]
    criterion_mse = criterion_list[2]
    criterion_sp = criterion_list[3]
    
    softmax = nn.Softmax(dim=1).cuda()
    

    model_s = module_list[0]
    model_s_fc_new = module_list[1]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    
    for idx, data in enumerate(train_loader):
  
        input, target, index = data
        
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
  
        # ===================forward=====================
        preact = False

        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        
        
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]
        
        fea_s  = feat_s[opt.hint_layer].detach()
        
        soft_t = softmax(logit_t/opt.net_T)
        

        #TODO: not needed when no interventions are used
        for i in range( len(target) ):
            context_new[target[i]] = context_new[target[i]]*( current_num[target[i]]/(current_num[target[i]]+soft_t[i][target[i]]) ) + fea_s[i]*(soft_t[i][target[i]]/(current_num[target[i]]+soft_t[i][target[i]]) )
            current_num[target[i]]+=soft_t[i][target[i]]
        
        #P.V: different to init train START

        p = softmax(logit_s.detach()/opt.net_T)

        #P.V: this is: a_i^s * \bar c_i        
        sam_contxt = torch.mm(p, context)
        

        #P.V: if we disable interventional loss, this should not get used as it uses context
        f_new = torch.cat((feat_s[opt.hint_layer], sam_contxt),1)
        
        logit_s_new = model_s_fc_new(f_new)
        
        #Cross Entropy between sample representation concatenated with context and target 
        loss_cls = criterion_cls(logit_s_new, target)
        #P.V: different to init train END
        
        loss_kl = criterion_kl(logit_s, logit_t)
        
        
        class_num = model_s_fc_new.linear.weight.shape[0]
        
        fea_reg = module_list[2]
        f_s = fea_reg(feat_s[opt.hint_layer])
        f_t = feat_t[opt.hint_layer]
            
        loss_sample = criterion_mse(f_s, f_t)
            
        list_s, list_t = cluster(feat_s[opt.hint_layer], f_t, target, class_num)
            
        involve_class = 0
            
        loss_class=0.0
            
        for k in range( len(list_s) ):
            
            cur_len = len( list_s[k] )
            
            if cur_len>=2:
                
                cur_f_s = torch.stack(list_s[k])
                cur_f_t = torch.stack(list_t[k])
                    
                loss_class+= criterion_sp(cur_f_s, cur_f_t) 
                    
                involve_class += 1
                    
                    
        if involve_class==0:
            loss_class = 0.0
        else:
            loss_class = loss_class/involve_class  

        #P.V: loss_cls + loss_kl => interventional loss
        #P.V: disabling interventional loss should just be removing loss_cls + loss_kl (same question as in init train tho)
        loss =  opt.aa*(loss_cls + loss_kl) + opt.bb * loss_sample + opt.cc * loss_class

        acc1, acc5 = accuracy(logit_s_new, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    
    
    context_new = opt.cu*context + (1-opt.cu)*context_new

    return top1.avg, losses.avg, context_new

def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_idx, data in enumerate(val_loader):
            input = data[0]
            target = data[1]
            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


        print(' * test Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg

def validate_st_no_int(val_loader, model, opt): 
    batch_time = AverageMeter()
    
    top1_new = AverageMeter()
    top5_new = AverageMeter()
    
    softmax = nn.Softmax(dim=1).cuda()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, data in enumerate(val_loader):
            input = data[0]
            target = data[1]
            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                

            # compute output
            _, output  = model(input, is_feat=True, preact=False)
            

            #TODO: not sure if this needs softmax too?
            output = softmax(output / opt.net_T)
            

            acc1_new, acc5_new = accuracy(output, target, topk=(1, 5))
            top1_new.update(acc1_new[0], input.size(0))
            top5_new.update(acc5_new[0], input.size(0))           
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print(' * Test Acc@1 {top1_new.avg:.3f} Acc@5 {top5_new.avg:.3f}'
              .format(top1_new=top1_new, top5_new=top5_new))
        
    return top1_new.avg, top5_new.avg



def validate_st(val_loader, model, criterion, opt, context, model_fc_new):
    """validation"""
    batch_time = AverageMeter()
    
    top1_new = AverageMeter()
    top5_new = AverageMeter()
    
    softmax = nn.Softmax(dim=1).cuda()

    # switch to evaluate mode
    model.eval()
    model_fc_new.eval()

    with torch.no_grad():
        end = time.time()
        for idx, data in enumerate(val_loader):
            input = data[0] 
            target = data[1]
            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                

            # compute output
            feat, output  = model(input, is_feat=True, preact=False)
            
            p = softmax(output/opt.net_T)
        
            sam_contxt = torch.mm(p, context)
        
            f_new = torch.cat((feat[opt.hint_layer], sam_contxt),1)
        
            output_new = model_fc_new(f_new)            
            

            acc1_new, acc5_new = accuracy(output_new, target, topk=(1, 5))
            top1_new.update(acc1_new[0], input.size(0))
            top5_new.update(acc5_new[0], input.size(0))           
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print(' * Test Acc@1 {top1_new.avg:.3f} Acc@5 {top5_new.avg:.3f}'
              .format(top1_new=top1_new, top5_new=top5_new))
        
    return top1_new.avg, top5_new.avg
