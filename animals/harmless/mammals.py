class Mammals:
    ''' Constructor for this class. '''
    def __init__(self):
        # Create some member animals
        self.members = ['Sheep', 'Pig', 'Wild cat']
    
    def printMembers(self):
        print('Printing harmless members of the Mammals class')
        for member in self.members:
            print('\t%s' % member)