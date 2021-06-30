import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# mean false error by Wang+16, IJCNN.
class MFE_Loss(nn.Module):
    def __init__(self, orig_loss):
        super().__init__()
        self.set_orig_loss(self, orig_loss)
        print("using MFE_Loss")

    def set_orig_loss(self, orig_loss):
        self.orig_loss = orig_loss

    def forward(self, inputs, targets):
        # print('%$%$%$ inputs.shape =', inputs.shape)
        # print('%$%$%$ targets.shape =', targets.shape)
        labelTypes = torch.unique_consecutive(targets)
        input_tensorDim = [0] + list(inputs.shape[1:])
        # print('input_tensorDim =', input_tensorDim)
        groupedInputs = [
            torch.empty(input_tensorDim, dtype=torch.float) for _ in labelTypes
        ]
        groupedTargets = [torch.empty((0), dtype=torch.long) for _ in labelTypes]
        # print('groupedInputs[0].shape =', groupedInputs[0].shape)
        # print('groupedTargets[0].shaoe =', groupedTargets[0].shape)
        label2ID = {label.item(): ID for ID, label in enumerate(labelTypes)}
        for input, target in zip(inputs, targets):
            input_reshaped = input.view((1, -1))
            target_tensor = torch.from_numpy(
                np.array([target], dtype=np.long), dtype=torch.long
            )
            groupedInputs[label2ID[target.item()]] = torch.cat(
                [groupedInputs[label2ID[target.item()]], input_reshaped]
            )

            groupedTargets[label2ID[target.item()]] = torch.cat(
                [groupedTargets[label2ID[target.item()]], target_tensor]
            )
            # print('input =', input)
            # print('target = ', target)
            # print('label2ID =', label2ID)
            # print('target =', target.item())
            # print('type(groupedInputs[label2ID[target.item()]]) =', type(groupedInputs[label2ID[target.item()]]))
            # print('type(input) =', type(input))
            # print('groupedInputs[label2ID[target.item()]].shape =', groupedInputs[label2ID[target.item()]].shape)
            # print('input.shape =', input.shape)
            ### print('input_reshaped.shape =', input_reshaped.shape)
            # print('target.shape =', target.shape)
            ### print('target_tensor.shape =', target_tensor.shape)
            # if input_reshaped.shape[0] != target_tensor.shape[0]:
            #    print('$$$$$$$$$')
            #    print('input_reshaped.shape[0] != target_tensor.shape[0]')
            #    print('input_reshaped.shape =', input_reshaped.shape)
            #    print('target_tensor.shape =', target_tensor.shape)
            # print('groupedTargets[label2ID[target.item()]].shape =', groupedTargets[label2ID[target.item()]].shape)
            # print('groupedInputs[label2ID[target.item()]].shape =', groupedInputs[label2ID[target.item()]].shape)
            # print('groupedTargets[label2ID[target.item()]].shaoe =', groupedTargets[label2ID[target.item()]].shape)
            def g2t(grouped, label):
                grouped[label2ID[label.item()]]

        return torch.as_tensor(
            F.sum(
                [
                    self.orig_loss(g2t(groupedInputs, lbl), g2t(groupedTargets, lbl))
                    / len(g2t(groupedInputs, lbl))
                    if len(g2t(groupedInputs, lbl)) > 0
                    else 0
                    for lbl in labelTypes
                ]
            )
        )


# mean squared false error by Wang+16, IJCNN.
class MSFE_Loss(torch.nn.Module):
    def __init__(self, orig_loss):
        super().__init__()
        self.set_orig_loss(orig_loss)

    def set_orig_loss(self, orig_loss):
        self.orig_loss = orig_loss

    def forward(self, x, y):
        #####
        assert False, "MSFE_Loss is not defined yet."
        # TODO "MSFE_Loss is not defined yet."
