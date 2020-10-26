import Denoised_method as dm
import function
from PIL import Image, ImageTk
from imp import reload
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import tkinter.filedialog


class App(tk.Tk) :
    def __init__(self) :
        tk.Tk.__init__(self)
        self.background = tk.Frame(master = self, bg = "gray5", width = 500, height = 500)
        self.background.grid_propagate(0)
        self.background.grid()
        self.file = 'noisdm.png'
        self.show_scene()
        self.__fig = Figure()

    def get_fig(self):
        return self.__fig

    def set_fig(self, i):
        self.__fig = i

    def show_scene(self):
        dt = tk.DoubleVar()
        dt.set(0.3)
        nb_it = tk.DoubleVar()
        nb_it.set(50)
        alpha = tk.DoubleVar()
        alpha.set(0.02)
        imFrame = tk.Frame(master = self.background, bg = "white")
        imFrame.grid(row = 1, column = 2)
        labelImg = tk.Label(master = imFrame, bg = "gray23")
        labelImg.grid(row = 30, column = 2)
        buttonframe = tk.Frame(master = self.background, bg = "gray5")
        buttonframe.grid(row = 0, column = 0)

        buttonframdm = tk.Frame(master = self.background, bg = "gray5")
        buttonframdm.grid(row = 50, column = 2)

        button_open = tk.Button(master = buttonframe, text = "Load", command = lambda : self.open_file(labelImg), bg = "gray9", fg = "white")
        button_open.grid(row = 0, column = 0)
        button_quit = tk.Button(master= buttonframe, text = 'Quit', command = self._quit, fg = "white", bg = "gray9")
        button_quit.grid(row = 0, column = 1)
        button_he= tk.Button(master = buttonframdm, text = 'Heat equation', command = lambda : self.save_param(dt, nb_it,1, alpha))
        button_he.grid(row = 0, column = 2)
        button_pm= tk.Button(master = buttonframdm, text = 'Perona_Malik', command = lambda : self.save_param(dt, nb_it,2, alpha))
        button_pm.grid(row = 0, column = 3)

        textFrame = tk.Frame(master = self.background, bg = "gray9")
        textFrame.grid(row = 1, column = 0)
        L3 = tk.Label(textFrame, text= "dt = ", width = 15, bg = "gray9", fg = "white")
        L3.grid(row = 7, column = 0)
        L4 = tk.Label(textFrame, text= "num of iterations : ", width = 15, bg = "gray9", fg = "white")
        L4.grid(row = 10, column = 0)
        L5= tk.Label(textFrame, text= " alpha= ", width = 15, bg = "gray9", fg = "white")
        L5.grid(row = 13, column = 0)

        E3 = tk.Entry(textFrame, textvariable =  dt, width = 5)
        E3.grid(row = 7, column = 1)
        E4 = tk.Entry(textFrame, textvariable = nb_it, width = 5)
        E4.grid(row = 10, column = 1)
        E5 = tk.Entry(textFrame, textvariable = alpha, width = 5)
        E5.grid(row = 13, column = 1)


    def open_file(self, label):
        filename = tk.filedialog.askopenfilename (filetypes = [("Images files","*.png")])
        self.file = filename
        im = Image.open(self.file)
        image = im.resize((250, 250), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo)
        label.image = photo
        #label.grid(row = 10, column = 4)



    def save_param(self,dt,nb_it, i, alpha) :
        labelImg = tk.Label(self.background, text= "image = ",  fg = "white", bg = "white")
        if i == 1:
            img = mpimg.imread(self.file)
        else :
            img = mpimg.imread(self.file)
        img1 = np.dot(img[...,:3], [int(0.299*255),int(0.587*255),int(0.114*255)])
        n_dt = dt.get()
        n_nb_it = nb_it.get()
        denoised = dm.Denoising_Methods(dt=n_dt,dx=1,dy=1,nb_it=n_nb_it)
        if i ==1 :
            self.__fig.clear()
            plt.close(self.__fig)
            fig1 = denoised.heat_equation(img1)
        else :
            self.__fig.clear()
            plt.close(self.__fig)
            (x,y) = np.shape(img1)
            fig1= denoised.perona_malik(img1, alpha = alpha.get(), fun= denoised.c_function)
        self.set_fig(fig1)

    def _quit(self) :
        self.quit()
        self.destroy()


app = App()
app.mainloop()
