import os

class itemList:
    def __init__(self, path) -> None:
        self.path = path
        folder_items = os.listdir(path)
        folder_items.sort()
        first_item = folder_items[0]
        second_item = folder_items[1]
        self.extension = first_item.split(".")[-1]
        self.extension_length = len(self.extension) + 1
        
        # Check if we have zero padding in the namming
        self.zero_padding = False
        self.padding_length = 0
        if first_item[0] == "0" and second_item[0] == "0": 
            self.zero_padding = True
            self.padding_length = len(first_item[:-self.extension_length])

        self.item_ids = []
        for item in folder_items:
            try:
                item_id = int(item[:-self.extension_length])
                self.item_ids.append(item_id)
            except:
                print(f"Item ({item[:-self.extension_length]}) cannot be converted to int. It will be discarded.")

        self.item_ids.sort()
        
    def getItemPath(self,idx:int):
        if self.zero_padding:
            path = os.path.join(self.path, str(self.item_ids[idx]).zfill(self.padding_length)+"." + self.extension)
        else:
            path = os.path.join(self.path, str(self.item_ids[idx])+"." + self.extension)
        return path
    
    def getItemPathFromName(self,name):
        path = os.path.join(self.path, name + "." + self.extension)
        return path
    
    def getItemID(self,idx):
        return self.item_ids[idx]
    
    def getItemName(self,idx):
        if self.zero_padding:
            name = str(self.item_ids[idx]).zfill(self.padding_length)
        else:
            name = str(self.item_ids[idx])
        return name

    def itemCount(self):
        return len(self.item_ids)