from PIL import Image

# 存储行图片的文件夹路径
row_folder = './pic_row/'

num_pic = 5
# 定义合并后的大图的宽度和高度
width = 4 * 300+30  # 假设每行图片都是384x384
height = num_pic * 384 - 70 * num_pic  # 假设有9行图片

# 创建一个空白图像
merged_image = Image.new('RGB', (width, height), color=(255, 255, 255))

for i in range(num_pic):
    row_image_file = row_folder + 'row_{}.png'.format(i)
    row_image = Image.open(row_image_file)
    # 裁剪图片
    if i != num_pic-1:
        crop_top = 20  # 上边界
        crop_bottom = row_image.height - 69  # 下边界（裁剪掉最后一段）
        row_image = row_image.crop((0, crop_top, row_image.width, crop_bottom))
    else:
        crop_top = 20
        crop_bottom = row_image.height
        row_image = row_image.crop((0, crop_top, row_image.width, crop_bottom))
    # 调整图片大小
    # row_image = row_image.resize((384, 384))
    merged_image.paste(row_image, (0, i * 300))

# 保存合并后的大图
merged_image.save('./merged_image.eps')

