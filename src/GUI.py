


############BROKEN###################




# from kivy.app import App
# from kivy.lang import Builder
# from kivy.uix.boxlayout import BoxLayout
# from kivy.core.window import Window
# from kivy.uix.floatlayout import FloatLayout
# from kivy.factory import Factory
# from kivy.properties import ObjectProperty
# from kivy.uix.popup import Popup
# import shutil
# import os
# import random
# import test
# 
# global image
# Builder.load_string('''
# <CameraClick>:
# 
#     BoxLayout:
#         orientation: 'vertical'
#         Camera:
#             id: camera
#             resolution: (640, 480)
#             play: False
#         Image:
#             id: image
#         ToggleButton:
#             text: 'Play video feed'
#             on_press: camera.play = not camera.play
#             size_hint_y: None
#             height: '48dp'
# 
#         BoxLayout:
#             size_hint_y: None
#             height: '48dp'
#             orientation: 'horizontal'
#             Button:
#                 text: 'Capture an image'
#                 size_hint_y: None
#                 height: '48dp'
#                 on_press: root.capture()
#             Button:
#                 text: 'Load an image'
#                 size_hint_y: None
#                 height: '48dp'
#                 on_press: root.show_load()
#         Label:
#             id: label1
#             text: 'Replicated emotion: Happiness'
#             size_hint_y: None
#             height: '48dp'
#         Label:
#             id: label2
#             text: 'Score: 83.8/100'
#             size_hint_y: None
#             height: '48dp'
# 
# <LoadDialog>:
#     BoxLayout:
#         size: root.size
#         pos: root.pos
#         orientation: "vertical"
#         FileChooserListView:
#             id: filechooser
# 
#         BoxLayout:
#             size_hint_y: None
#             height: 30
#             Button:
#                 text: "Cancel"
#                 on_release: root.cancel()
# 
#             Button:
#                 text: "Load"
#                 on_release: root.load(filechooser.path, filechooser.selection)
# 
# ''')
# 
# 
# class LoadDialog(FloatLayout):
#     load = ObjectProperty(None)
#     cancel = ObjectProperty(None)
# 
# 
# class CameraClick(BoxLayout):
#     emotions = ['Happiness', 'Anger', 'Fear', 'Neutral', 'Sadness', 'Surprise', 'Contempt']
#     score = 0
#     emoji = 0
# 
#     def dismiss_popup(self):
#         self._popup.dismiss()
# 
#     def capture(self):
#         '''
#         Function to capture the images and give them the names
#         according to their captured time and date.
#         '''
#         emotion_label = self.ids['label1']
#         img_canvas = self.ids['image']
#         camera = self.ids['camera']
#         camera.export_to_png("img/IMG_{}.png".format(emotion_label.text))
#         print("Captured")
# 
#         img_canvas.source = 'img/IMG_{}.png'.format(self.emotions[self.emoji])
#         image = img_canvas.source
#         test.get_image(image)
#         self.call_facial_recognition()
#         self.emoji = random.randint(0, 6)
#         emotion_label.text = str(self.emotions[self.emoji])
# 
#     def load(self, file_path, file):
#         '''
#         Function to load an image to the AI
#         '''
#         emotion_label = self.ids['label1']
# 
#         print(file_path)
#         print(file)
#         shutil.copy(file[0], 'img/')
# 
#         dst_file = os.path.join('img/', os.path.basename(file[0]))
#         print(dst_file)
#         new_dst_file_name = os.path.join('img/', 'IMG_{}.png'.format(emotion_label.text))
#         os.rename(dst_file, new_dst_file_name)
#         self.dismiss_popup()
#         self.call_facial_recognition()
# 
#         img_canvas = self.ids['image']
#         img_canvas.source = 'img/IMG_{}.png'.format(self.emotions[self.emoji])
#         image = img_canvas.source
#         test.get_image(image)
#         self.emoji = random.randint(0, 6)
#         emotion_label.text = str(self.emotions[self.emoji])
# 
#     def show_load(self):
#         content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
#         self._popup = Popup(title="Load file", content=content,
#                             size_hint=(0.9, 0.9))
#         self._popup.open()
# 
#     def call_facial_recognition(self):
#         score_label = self.ids['label2']
#         this_score = test.classification
#         self.score += this_score
#         score_label.text = str(self.score)
# 
# 
# class TestCamera(App):
#     def build(self):
#         Window.bind(on_dropfile=self._on_file_drop)
#         return CameraClick()
# 
#     def _on_file_drop(self, window, file_path):
#         print(file_path)
#         return
# 
# 
# TestCamera().run()
