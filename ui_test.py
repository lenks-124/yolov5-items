import sys,os
import cv2.cv2
from PyQt5.QtWidgets import QApplication, QMainWindow,QWidget,QFileDialog,QLabel
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage,QPixmap,QIcon,QPalette,QBrush,QColor
from det_gui import *
import cv2
from yolov5 import detect as dt
import threading
from pathlib import Path
import yolov5.val_2 as vl2
opts = {'source':r'D:\DeepLearning\projects\Item_YOLO\data\images', # detect时的source 可以为文件,目录,webcam
        'default_path':r'D:\DeepLearning\projects\Item_YOLO\data',  # 选择图片时默认打开的位置
        'data_yaml':r'D:\DeepLearning\projects\Item_YOLO\yolov5\models\guanliu.yaml', # 标签的yaml文件
        'img_single_path':'',  # 单张图片的读取地址    --待弃用
        'img_single_result':'', # 单张图片的detect地址
        'imgs_path':'',  # 批量检测 选择目录时所选中的地址    --待弃用
        'imgs_result_path':'',  # 批量检测的 结果地址
        'save_path_base':r'D:\DeepLearning\projects\Item_YOLO\data\results', # detect时保存结果的地址 默认是runs/detect 等
        'save_dir_base':'', # 真正检测结果的保存地址。修改为run函数返回 目前是 data/results/exp2
        'img':None, # 读到的一张图 cv2格式
        'img_result':None, # 上面图的检测结果 cv2格式
        'mode' : 1, # 当前模式 检测模式还是评估模式
        'flag':True,# 是否通过
        }
class MyWin(QWidget,Ui_Form):

    def __init__(self):
        super(MyWin,self).__init__()
        self.setupUi(self)
        self.label_12.setPixmap(QPixmap("logo_blue.png"))
        self.label_12.setScaledContents(True)
        self.label.setPixmap(QPixmap("img.png"))
        self.label2.setPixmap(QPixmap("success.png"))
        op1 = QtWidgets.QGraphicsOpacityEffect() # 将两个在画图label 上的 按钮 变透明
        op1.setOpacity(0)
        self.get_image.setGraphicsEffect(op1)
        op2 = QtWidgets.QGraphicsOpacityEffect()
        op2.setOpacity(0)
        self.result_img.setGraphicsEffect(op2)
        self.img = None
        self.result = None
        self.setWindowIcon(QIcon('logo.jpg'))
        self.directory = None

        # self.timer_camera1 = QTimer()   #
        # self.timer_camera2 = QTimer()
        # # 打开视频文件

        self.btn1_1.clicked.connect(self.selectImage)   # 选择图片
        self.get_image.clicked.connect(self.selectImage)  # 选择图片
        self.btn1_2.clicked.connect(self.decImage_item)      # 检测图片
        self.btn1_3.clicked.connect(self.openPic) # 自带查看器打开图片
        self.result_img.clicked.connect(self.openPic)  # 选择图片
        # self.btn2_1.clicked.connect(self.selectVideo)   # 选择视频
        # self.btn2_2.clicked.connect(self.decVideo)      # 检测视频
        # self.btn3_1.clicked.connect(self.selectCamera)  # 开启摄像头
        # self.btn3_2.clicked.connect(self.decVideo)      # 摄像头检测
        self.btn4_1.clicked.connect(self.mode_change)   # 模式切换
        self.btn5_1.clicked.connect(self.detect_pics)   # 图片批量检测
        self.btn5_2.clicked.connect(self.pics) # 打开文件目录
        self.btn5_3.clicked.connect(self.openDir) # 资源管理器中显示



        self.tps.append("<font>数据初始化成功...</font> </br>")
        self.tps.append("<font color='red'>采用CUDA加速图像处理...</font> </br>")
        self.tps.append("YOLOv5l torch 1.8.1 CUDA:0 (GeForce GTX 1650, 4096MiB)")
        self.tps.append("Fusing layers... ")
        self.tps.append("Model Summary: 468 layers, 46165219 parameters, 0 gradients, 108.0 GFLOPs")



    ## 模型重载的函数
    # def modelLoad(self):
    #     # self.tmp = test.Tester(weight_path=self.opt.weight_path)
    #     # self.tps.append('YOLOv3模块加载成功...')
    #     save_path = dt.run(source=opts['img_single_path'], data=opts['data_yaml'], device='cuda:0',
    #                        project=opts['save_path'])
    #     # img_result_path = opts['save_path'] + '/' + opts['img_single_path'].split('/')[-1]
    #     self.paintPic(cv2.imread(save_path))


    def openPic(self):
        t = threading.Thread(target=os.system,args=(opts['img_single_result'],))
        t.start()
        pass

    # 将cv2格式图片显示到label中
    def paintPic(self,img,label):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 不加的话，会变色

        width,height = img.shape[1],img.shape[0]

        # 设置新的图片分辨率框架
        width_new,height_new= label.width(),label.height()

        # print(width / height, width_new / height_new)

        # 判断并根据标签重设图片的长宽比率
        if width / height >= width_new / height_new:
            img = cv2.resize(img, (width_new, int(height * width_new / width)))
        else:
            img = cv2.resize(img, (int(width * height_new / height), height_new))


        img = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1],
                     QtGui.QImage.Format_RGB888)  # 这里必须是img.data 参数是地址

        pix = QPixmap(img)  # .scaled(self.label.width(),self.label.height()) # 如果不加scaled,只会显示部分图像
        label.setPixmap(pix)



    ##  选择图片的函数
    def selectImage(self):
        self.flag.lower()
        img_path,imgType= QFileDialog.getOpenFileName(self, '选取图片', opts['default_path'], "*.jpg;;All Files(*)")
        if img_path:
            opts['img_single_path'] = img_path
            print(img_path)
            self.img = cv2.imread(img_path)
            self.paintPic(self.img,self.label)
            self.tps.append(opts['img_single_path'] + ' 加载成功...')
        else:
            self.tps.append('请重新选择图片')

    ## 此模型的检测图片的函数
    def decImage_item(self):
        res, ans = dt.run(source=opts['img_single_path'], data=opts['data_yaml'], project=opts['save_path_base'])
        self.t1.setText(str(res[0]))
        self.t2.setText(str(res[1]))
        self.t3.setText(str(res[2]))
        self.t4.setText(str(res[3]))
        self.t5.setText(str(res[4]))
        self.t6.setText(str(res[5]))
        self.t7.setText(str(res[:6].sum()))
        print("-------")

        opts['img_single_result'] = ans['save_path']
        self.tps.append("结果图路径: " + opts['img_single_result'])
        opts['img_result'] = cv2.imread(ans['save_path'])
        self.paintPic(opts['img_result'], self.label2)

        if opts['mode'] == 1:
            if res[:6].sum() == 0:
                opts['flag'] = True
                self.t8.setText("正常")
            else :
                self.t8.setText("缺陷")
                opts['flag'] = False
            self.t9.setText(str(res[6]/(res[6]+res[7]) * 100)+"%")
            self.flag.raise_()
            if opts['flag']:
                png = QtGui.QPixmap("yes.png")
                self.flag.setPixmap(png)
                # self.paintPic(cv2.imread("yes.png"), self.flag)
            else:
                # self.paintPic(cv2.imread("no.png"), self.flag)
                png = QtGui.QPixmap("no.png")
                self.flag.setPixmap(png)


        if opts['mode'] == 0:
            print(opts['img_single_path'],Path(ans['save_path']).parent)
            a , b = vl2.count_tp(opts['img_single_path'],Path(ans['save_path']).parent)

            self.t8.setText(str(b))
            self.t9.setText(str(a))

    ## 选择目录
    def pics(self):
        self.flag.lower()
        opts['imgs_path'] = QFileDialog.getExistingDirectory()
        if opts['imgs_path']:
            print(opts['imgs_path'])
            self.tps.append("已选中" + opts['imgs_path'] + "文件夹...")
        else:
            self.tps.append("请重新选择文件夹...")

    ## 批量图片检测
    def detect_pics(self):
        # 加载第一张图并显示第一张图的结果
        first_img = os.listdir(opts['imgs_path'])[0]
        print("该文件夹下第一张图: " + first_img)
        opts['img'] = cv2.imread(os.path.join(opts['imgs_path'],first_img))
        self.paintPic(opts['img'],self.label)
        res, ans = dt.run(source=opts['imgs_path'], data=opts['data_yaml'], project=opts['save_path_base'])

        self.t1.setText(str(res[0]))
        self.t2.setText(str(res[1]))
        self.t3.setText(str(res[2]))
        self.t4.setText(str(res[3]))
        self.t5.setText(str(res[4]))
        self.t6.setText(str(res[5]))
        self.t7.setText(str(res[:6].sum()))
        if opts['mode'] == 1:
            self.t8.setText("--")
        self.t9.setText(str(int(res[6] / (res[6] + res[7]) * 100)) + '%')
        opts['imgs_result_path'] = str(ans['save_dir_base'])
        self.tps.append("图片测试结果路径: " )
        self.tps.append(opts['imgs_result_path'])

        opts['img_result'] = cv2.imread(os.path.join(ans['save_dir_base'], first_img))
        self.paintPic(opts['img_result'], self.label2)



    ## 资源管理器中显示
    def openDir(self):
        os.startfile(opts['imgs_result_path'])
        return


    def test(self):
        pass



    def mode_change(self):
        print(self.comboBox4.currentText())
        opts['mode'] = 1 if '检测' in self.comboBox4.currentText() else 0
        print(opts['mode'])
        if opts['mode'] == 1:
            self.label_10.setText("结果:")
            self.label_11.setText("合格率:")
        else:
            self.label_10.setText("漏检个数:")
            self.label_11.setText("误检个数:")
        self.set_textedit_0()
    def set_textedit_0(self):
        self.t1.setText("")
        self.t2.setText("")
        self.t3.setText("")
        self.t4.setText("")
        self.t5.setText("")
        self.t6.setText("")
        self.t7.setText("")
        self.t8.setText("")
        self.t9.setText("")
if __name__ == '__main__':
    app = QApplication(sys.argv)
    wn = MyWin()
    wn.setWindowIcon(QIcon("logo_red.png"))
    wn.show()
    sys.exit(app.exec_())