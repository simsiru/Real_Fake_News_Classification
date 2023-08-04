import torch


def process_output(output_tensor: torch.tensor) -> list:
    pred = []

    for pred_batch in output_tensor:
        batch_list = (torch.argmax(pred_batch[1], dim=1)).int().tolist()

        for label in batch_list:
            pred.append(label)
    
    return pred