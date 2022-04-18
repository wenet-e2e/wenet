# authored by richardt@graphcore.ai
import torch
import poptorch

INDEX_PREFIX = '__'
SPLIT_SYMBOL = '.'


def lfind(string, symbol=SPLIT_SYMBOL):
    index = string[::-1].rfind(symbol)
    index = len(string) - index
    return index


def split_layer_path(layer_path, symbol=SPLIT_SYMBOL):
    index = lfind(layer_path, symbol)
    current_layer_name = layer_path[: index-1]
    child_layer_path = layer_path[index:]
    return current_layer_name, child_layer_path


class BasePipelineModel(torch.nn.Module):
    """
    base pipeline model.
    useage inherite this class just as a nn.Module
    """

    def set_start_point_list(self, child_layer_path_list):
        # child_layer_path_list: [('path1', 1), ('path2', 2)]
        for path, ipu_id in child_layer_path_list:
            self.set_start_point(path, ipu_id)

    def set_start_point(self, child_layer_path, ipu_id=0):
        self._nested_set_start_point(layer_pointer=self, child_layer_path=child_layer_path, ipu_id=ipu_id)

    def _nested_set_start_point(self, layer_pointer, child_layer_path, ipu_id):
        current_layer_name, child_layer_path = split_layer_path(layer_path=child_layer_path)

        if INDEX_PREFIX in current_layer_name:
            current_layer_index = int(current_layer_name.split(INDEX_PREFIX)[-1])
            current_layer_name = current_layer_name.split(INDEX_PREFIX)[0]
        else:
            current_layer_index = None

        if current_layer_index is not None:
            layer_pointer = layer_pointer.__getattr__(current_layer_name)
            sub_layer_pointer = layer_pointer[current_layer_index]
            if child_layer_path == "":
                layer_pointer[current_layer_index] = self._warp_start_point(sub_layer_pointer, ipu_id=ipu_id)
                return
            else:
                self._nested_set_start_point(sub_layer_pointer, child_layer_path, ipu_id=ipu_id)
        else:
            sub_layer_pointer = layer_pointer.__getattr__(current_layer_name)
            if child_layer_path == "":
                layer_pointer.__setattr__(current_layer_name, self._warp_start_point(sub_layer_pointer, ipu_id=ipu_id))
                return
            else:
                self._nested_set_start_point(sub_layer_pointer, child_layer_path, ipu_id=ipu_id)

    def _warp_start_point(self, layer_pointer, ipu_id=0):
        layer_pointer = poptorch.BeginBlock(layer_pointer, ipu_id=ipu_id)
        return layer_pointer
