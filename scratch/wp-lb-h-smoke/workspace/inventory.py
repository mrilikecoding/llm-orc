class Inventory:
    def __init__(self):
        self.items = {}

    def add_item(self, name, quantity):
        if name in self.items:
            self.items[name] += quantity
        else:
            self.items[name] = quantity

    def remove_item(self, name, quantity):
        if name in self.items:
            self.items[name] -= quantity
            if self.items[name] <= 0:
                del self.items[name]

    def total_value(self):
        return sum(self.items.values())

if __name__ == "__main__":
    inventory = Inventory()
    inventory.add_item("apple", 5)
    inventory.add_item("banana", 3)
    inventory.remove_item("apple", 2)
    print("Total items:", inventory.total_value())