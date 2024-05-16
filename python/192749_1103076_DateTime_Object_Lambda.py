import datetime as dt
import time as tm

tm.time()

dtnow = dt.datetime.fromtimestamp(tm.time())
dtnow

dtnow.year, dtnow.month, dtnow.day, dtnow.hour, dtnow.minute, dtnow.second # get year, month, day, etc.from a datetime

delta = dt.timedelta(days = 100) # create a timedelta of 100 days
delta

today = dt.date.today()

today - delta # the date 100 days ago

today > today-delta # compare dates

class Person:
    department = 'School of Information' #a class variable

    def set_name(self, new_name): #a method
        self.name = new_name
    def set_location(self, new_location):
        self.location = new_location

person = Person()
person.set_name('Christopher Brooks')
person.set_location('Ann Arbor, MI, USA')
print('{} live in {} and works in the department {}'.format(person.name, person.location, person.department))

store1 = [10.00, 11.00, 12.34, 2.34]
store2 = [9.00, 11.10, 12.34, 2.01]
cheapest = map(min, store1, store2)
cheapest

for item in cheapest:
    print(item)

my_function = lambda a, b, c : a + b

my_function(1, 2, 3)

my_list = [number for number in range(0,1000) if number % 2 == 0]

