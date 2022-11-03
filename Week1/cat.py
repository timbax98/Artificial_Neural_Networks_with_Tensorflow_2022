class Cat:
    def __init__(self, name):
        self.name = name

    def introduce_self(self):
        print("My name is " + self.name + " you must be " + c2.name)

c1= Cat(input("Enter Name: "))
c2= Cat(input("Enter Name: "))

c1.introduce_self()
