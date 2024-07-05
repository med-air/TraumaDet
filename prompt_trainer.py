import os
import time 

import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import torch.nn as nn

def cls_score(pred, label):
    """
    pred:  [[1, 0, 0, 0], [0, 0, 0, 1]]
    label: [[0, 0, 1, 1], [1, 1, 1, 1]]
    """

    # case level
    case_pred = np.array([np.any(item) for item in pred], dtype=int)
    case_true = np.array([np.any(item) for item in label], dtype=int)
    tn, fp, fn, tp = confusion_matrix(case_true, case_pred).ravel()

    case_acc = np.sum(case_pred == case_true) / (len(case_true) + 1e-9)
    case_sen = tp / (tp + fn + 1e-9)
    case_prec = tp / (tp + fp + 1e-9)
    case_f1  = (2*case_prec*case_sen) / (case_prec + case_sen + 1e-9)


    # organ level
    organ_pred = np.array(pred).ravel()
    organ_true = np.array(label).ravel()
    tn, fp, fn, tp = confusion_matrix(organ_true, organ_pred).ravel()
    
    organ_acc = np.sum(organ_pred == organ_true) / (len(organ_true) + 1e-9)
    organ_sen = tp / (tp + fn + 1e-9)
    organ_prec = tp / (tp + fp + 1e-9)
    organ_f1   = (2*organ_prec*organ_sen) / (organ_prec + organ_sen + 1e-9)


    score_table = {
        "case_acc": case_acc,
        "case_sensitive": case_sen,
        "case_precision": case_prec,
        "case_f1": case_f1,
        "organ_acc": organ_acc,
        "organ_sensitive": organ_sen,
        "organ_precision": organ_prec,
        "organ_f1": organ_f1
    }
    score = 0.3 * case_sen + 0.2 * case_f1 + 0.1 * case_acc + 0.2 * organ_acc + 0.2 * organ_f1

    return score, score_table

def get_Trauma_embedding(labels):
    Truama_embedding = torch.load("Trauma_Embeding.pth")
    # labels shape: B, 3
    B = labels.shape[0]
    batch_kl_features = []
    for b in range(B):

        f_liver  = Truama_embedding[(labels[b,0].int()),...]
        f_spleen = Truama_embedding[((labels[b,1]+2).int()),...]
        f_kidney = Truama_embedding[((labels[b,2]+4).int()),...]
        b_feature = torch.cat((f_liver, f_spleen, f_kidney), axis=0)
        batch_kl_features.append(b_feature)

    batch_kl_features = torch.stack(batch_kl_features)
    
    return batch_kl_features

def get_global_prompt(labels):
    path = "./Trauma_Label.pth"
    result = torch.load(path)
    prompt_template = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

    global_prompts = []
    for array in labels:
        index = prompt_template.index(array.tolist())
        global_prompts.append(result[index].cpu())

    # 将列表转换为NumPy数组，并沿着新的轴维度堆叠
    global_prompts = np.stack(global_prompts, axis=0)
    global_prompts = torch.from_numpy(global_prompts)

    return global_prompts

def trainer(model, train_loader, val_loader, optimizer, scheduler, loss_function, prompt_loss, args):
    alfa = args.alfa
    device = args.device    
    save_root = os.path.join(args.log_dir, args.model_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    best_metric = -1
    for epoch in range(args.max_epochs):
        scheduler.step()
        epoch_time = time.time()
        ## Training
        print("-" * 10)
        model.train()
        epoch_loss = 0
        step = 0
        start_time = time.time()
        for batch_data in train_loader:
            step += 1
            liver     = batch_data['liver'].to(device)
            spleen    = batch_data['spleen'].to(device)
            left_kidney = batch_data['left_kidney'].to(device)
            right_kidney = batch_data['right_kidney'].to(device)
            
            abdominal = batch_data['abdominal'].to(device)

            labels  = batch_data['label'].to(device)

            gloabl_prompts = get_global_prompt(labels).cuda()
            Truama_embedding = get_Trauma_embedding(labels).cuda()

            optimizer.zero_grad()

            if args.model_name == "local_prompt_global_prompt_singleFusion_embedding":
                
                model.trauma_embedding.data = Truama_embedding.float()
                outputs, alignfeature, weights_feature = model(abdominal, liver, spleen, left_kidney, right_kidney, isTraining = True)
                prompt_loss = True
                CEloss = loss_function(outputs, labels)
                kl_input = F.log_softmax(alignfeature, dim=1)
                kl_target = F.log_softmax(weights_feature, dim=1)
                kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
                KLloss = kl_loss(kl_input, kl_target)
            
            elif args.model_name == "Global_Prompt":
                outputs, alignfeature = model(abdominal)
                CEloss = loss_function(outputs, labels)
                if prompt_loss:
                    kl_input = F.log_softmax(alignfeature, dim=1)
                    kl_target = F.log_softmax(gloabl_prompts, dim=1)
                    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
                    KLloss = kl_loss(kl_input, kl_target)
                else:
                    KLloss = 0

            else:
                outputs, alignfeature = model(abdominal, liver, spleen, left_kidney, right_kidney)
                CEloss = loss_function(outputs, labels)

                if prompt_loss:
                    kl_input = F.log_softmax(alignfeature, dim=1)
                    kl_target = F.log_softmax(gloabl_prompts, dim=1)
                    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
                    KLloss = kl_loss(kl_input, kl_target)
                else:
                    KLloss = 0


            CEloss = loss_function(outputs, labels)
            loss = alfa * CEloss + (1 - alfa) * KLloss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            outputs  = outputs.detach().cpu().numpy()
            labels   = labels.detach().cpu().numpy()

            if prompt_loss:
                print('Epoch {}/{} {}/{}'.format(epoch + 1, args.max_epochs, step, len(train_loader)),
                    'loss: {:.4f}'.format(loss.item()),
                    'CE: {:.4f}'.format(CEloss.item()),
                    'KL: {:.4f}'.format(KLloss.item()),
                    'time {:.2f}s'.format(time.time() - start_time))
                with open(os.path.join(save_root, 'log.txt'), 'a') as f:
                    print('Epoch {}/{} {}/{}'.format(epoch + 1, args.max_epochs, step, len(train_loader)),
                        "outputs: ", outputs.ravel(),
                        "labels: ", labels.ravel(),
                        'loss: {:.4f}'.format(loss.item()),
                        'CE: {:.4f}'.format(CEloss.item()),
                        'KL: {:.4f}'.format(KLloss.item()),
                        'time {:.2f}s'.format(time.time() - start_time), file=f)
            else:
                print('Epoch {}/{} {}/{}'.format(epoch + 1, args.max_epochs, step, len(train_loader)),
                    'loss: {:.4f}'.format(loss.item()), 
                    'time {:.2f}s'.format(time.time() - start_time))
                with open(os.path.join(save_root, 'log.txt'), 'a') as f:
                    print('Epoch {}/{} {}/{}'.format(epoch + 1, args.max_epochs, step, len(train_loader)),
                        "outputs: ", outputs.ravel(),
                        "labels: ", labels.ravel(),
                        'loss: {:.4f}'.format(loss.item()),
                        'time {:.2f}s'.format(time.time() - start_time), file=f)
            start_time = time.time()

        epoch_loss /= step
        print('Final training  {}/{}'.format(epoch + 1, args.max_epochs), 'loss: {:.4f}'.format(epoch_loss),
                  'time {:.2f}s'.format(time.time() - epoch_time))
        with open(os.path.join(save_root, 'log.txt'), 'a') as f:
            print('Final training  {}/{}'.format(epoch + 1, args.max_epochs), 'loss: {:.4f}'.format(epoch_loss),
                  'time {:.2f}s'.format(time.time() - epoch_time), file=f)
        
        # save the model
        torch.save(model.state_dict(), os.path.join(save_root, 'model_last.pt'))

        ## Evaluation
        if (epoch + 1) % args.val_every == 0:
            torch.save(model.state_dict(), os.path.join(save_root, f'model_{epoch}.pt'))

            print("-" * 20)
            print("Start to Evaluation")
            y_true = []
            y_pred = []
            eval_time = time.time()
            with torch.no_grad():
                model.eval()
                start_time = time.time()
                for val_data in val_loader:
                    val_liver        = val_data['liver'].to(device)
                    val_spleen        = val_data['spleen'].to(device)
                    val_left_kidney  = val_data['left_kidney'].to(device)
                    val_right_kidney = val_data['right_kidney'].to(device)

                    val_abdominal = val_data['abdominal'].to(device)
                    

                    # val_segs   = val_data['seg'].to(device)
                    val_labels  = val_data['label'].to(device)
                    names   = val_data['name']

                    # # # apply val_segs to val_inputs
                    # val_inputs = val_inputs * (val_segs > 0.5)
                    if args.model_name == "Global_Prompt":
                        val_preds, _ = model(val_abdominal)
                    else:
                        val_preds, _ = model(val_abdominal, val_liver, val_spleen, val_left_kidney, val_right_kidney)

                    # val_preds = model(val_abdominal, val_liver, val_spleen, val_left_kidney, val_right_kidney)
                    val_preds[val_preds >= 0] = 1
                    val_preds[val_preds < 0] = 0
                    val_labels = val_labels.cpu().numpy()
                    val_preds  = val_preds.detach().cpu().numpy()
                    y_true.extend(val_labels)
                    y_pred.extend(val_preds)
                    print("names", names, "val labels:",val_labels, "preds:", val_preds, 'time {:.2f}s'.format(time.time() - start_time))
                    with open(os.path.join(save_root, 'log.txt'), 'a') as f:
                        print("names", names, "val labels:", val_labels, "preds:", val_preds, 'time {:.2f}s'.format(time.time() - start_time), file=f)
                    start_time = time.time()

            score, score_table = cls_score(y_pred, y_true)
            metrics = score
            print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1), "Metrics", metrics, 'Score', score, "Organ F1", score_table['organ_f1'], 'time {:.2f}s'.format(time.time() - eval_time))
            with open(os.path.join(save_root, 'log.txt'), 'a') as f:
                print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1), "Metrics", metrics, 'Score', score, "Organ F1", score_table['organ_f1'], 'time {:.2f}s'.format(time.time() - eval_time), file=f)
                print(score_table, file=f)
            if metrics > best_metric:
                best_metric = metrics
                torch.save(model.state_dict(), os.path.join(save_root, 'model_best.pt'))
                print("saved new best metric model!")



if __name__ == "__main__":
    y_true = [np.array([1, 1, 0, 0]), np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])]
    y_pred = [np.array([1, 0, 1, 0]), np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]), np.array([0, 0, 0, 1])]

    score, score_table = cls_score(y_pred, y_true)
    print(score)
    print(score_table)