class Birds:
    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Ostrich', 'Emu', 'Lammergeier']
    
    def printMembers(self):
        print('Printing dangerous members of the Birds class')
        for member in self.members:
            print('\t%s' % member)