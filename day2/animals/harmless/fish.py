class Fish:
    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Trout', 'Salmon', 'Pike']
    
    def printMembers(self):
        print('Printing harmless members of the Fish class')
        for member in self.members:
            print('\t%s' % member)