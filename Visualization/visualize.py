"""
This program visualize the generated trace on the given game.
"""
from PIL import Image


class GambleVisualize(object):
    def __init__(self):
        self.agent_2 = Image.open('.\\Visualization\\FairGamble\\Person.png').resize((90,90))
        self.agent_1 = Image.open('.\\Visualization\\FairGamble\\Person.png').resize((90,90))
        self.agent_channels = self.agent_1.split()
        self.background = Image.open('.\\Visualization\\FairGamble\\Background.png').resize((800,600))
        self.anime = []
    def reset(self):
        self.background.paste(self.agent_1,(10,50),mask=self.agent_channels[3])
        self.background.paste(self.agent_1,(10,50),mask=self.agent_channels[3])
        self.anime.append(self.background.copy())
    
    def process(self,):


vis = GambleVisualize()
vis.reset()
