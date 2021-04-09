from abc import ABC, abstractmethod

class vehicle(ABC):
    "This class inherits from ABC"
    @abstractmethod
    def number_of_weels(self):
        '''This method is abstract, so the class cannot be
           instantiated. This method will be overridden in
           subclasses of Vehicle.'''
        pass
    @abstractmethod
    def number_of_seats(self):
        '''Another abstract mehtod'''
        pass

class car(vehicle):
    def number_of_weels(self):
        return 4
    def number_of_seats(self):
       return 3


c=car()
c.number_of_weels()
c.number_of_seats()

'''

Important notes:

1. So abstract classes or base classes are used to ensure
the abstract methods are used in parent class or derived class,
but the implementation is provided in the parent class, not in the abract class.Very importantly
those methods can provide different functionality in the derived
class, not nessesarily providing the same functinality

2. we cann't create objects using the parent class if all
methodes defined in abracts class not present in parent class

'''