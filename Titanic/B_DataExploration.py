class DataExploration:

    def __init__(self, data):
        self.dataset = data

    def describe (self):

        print(self.dataset.describe())


    def charts (self):
        import matplotlib.pyplot as plt
        import seaborn as sns

        _,ax = plt.subplots(3,4,figsize=(8,5))

        ax[0,0].set_title('Total Passengers by Class')
        sns.countplot('Pclass',data=self.dataset,ax=ax[0,0])
        ax[0,1].set_title('Total Passengers by Gender')
        sns.countplot('Sex',data=self.dataset,ax=ax[0,1])
        ax[0,2].set_title('Age Box Plot By Class')
        sns.boxplot(x='Pclass',y='Age',data=self.dataset,ax=ax[0,2])
        ax[0,3].set_title('Survival Rate by SibSp')
        sns.countplot('SibSp',hue='Survived',data=self.dataset,ax=ax[0,3],palette='husl')
        ax[1,0].set_title('Survival Rate by Class')
        sns.countplot('Pclass',hue='Survived',data=self.dataset,ax=ax[1,0],palette='husl')
        ax[1,1].set_title('Survival Rate by Gender')
        sns.countplot('Sex',hue='Survived',data=self.dataset,ax=ax[1,1],palette='husl')
        ax[1,2].set_title('Survival Rate by Age')
        sns.distplot(self.dataset[self.dataset['Survived']==0]['Age'].dropna(),ax=ax[1,2],kde=False,color='r',bins=5)
        sns.distplot(self.dataset[self.dataset['Survived']==1]['Age'].dropna(),ax=ax[1,2],kde=False,color='g',bins=5)
        ax[1,3].set_title('Survival Rate by Parch')
        sns.countplot('Parch',hue='Survived',data=self.dataset,ax=ax[1,3],palette='husl')
        ax[2,0].set_title('Fare Distribution')
        sns.distplot(self.dataset['Fare'].dropna(),ax=ax[2,0],kde=False,color='b')
        ax[2,1].set_title('Survival Rate by Fare and Pclass')
        sns.swarmplot(x='Pclass',y='Fare',hue='Survived',data=self.dataset,palette='husl',ax=ax[2,1])
        ax[2,2].set_title('Total Passengers by Embarked')
        sns.countplot('Embarked',data=self.dataset,ax=ax[2,2])
        ax[2,3].set_title('Survival Rate by Embarked')
        sns.countplot('Embarked',hue='Survived',data=self.dataset,ax=ax[2,3],palette='husl')

        plt.show()

    def sexAnalysis(self):

        groupBy = self.dataset.groupby(['Sex','Survived']).size().unstack()
        print(groupBy)

        import matplotlib.pyplot as plt
        groupBy.plot(kind='bar').set_title('Gender histogram training data')
        plt.show()

    def pclass(self):
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.countplot('Pclass',hue='Survived',data=self.dataset)
        plt.show()

    def head(self, num=5):
        # First X rows
        print( self.dataset.head(num) ) 

    def plot_correlation_map(self):
        import seaborn as sns
        import matplotlib.pyplot as plt
        corr = self.dataset.corr()
        _ , ax = plt.subplots( figsize =( 12 , 10 ) )
        cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
        _ = sns.heatmap(
            corr, 
            cmap = cmap,
            square=True, 
            cbar_kws={ 'shrink' : .9 }, 
            ax=ax, 
            annot = True, 
            annot_kws = { 'fontsize' : 12 }
        )
        plt.show()