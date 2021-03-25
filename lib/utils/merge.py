import torch


def merge_template_search(inp_list):
    return {"feat": torch.cat([x["feat"] for x in inp_list], dim=0),
            "mask": torch.cat([x["mask"] for x in inp_list], dim=1),
            "pos": torch.cat([x["pos"] for x in inp_list], dim=0)}
