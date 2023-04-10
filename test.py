"""
	Import necessary libraries
"""
from cs1media import *
import time

def downsample(input_image, rescaling_factor):
    """
        Implement a function that outputs a downsampled version of the input image.

    Args:
        input_image:        image to be downsampled [cs1media Picture object]
        rescaling_factor:   downsampling factor [int]

    Returns:
        downsampled_image:  downsampled image [cs1media Picture object]

    """
    # -----
    # TODO: Write your code here
    w,h = input_image.size()
    if h%rescaling_factor == 0 and w%rescaling_factor == 0:
        neww= w//rescaling_factor
        newh = h//rescaling_factor
        new_image = create_picture(neww,newh)
        for y in range(newh):
            for x in range(neww):
                #0,0 0,1 1,0 1,1 -> 0,0
                #2,0 2,1 3,0 3,1 -> 1,0
                #2,2 2,3 3,2 3,3 -> 1,1

                #0,0~0,3 / 0,0~3,0
                rs = []
                gs = []
                bs = []
                for i in range(rescaling_factor):
                    for j in range(rescaling_factor):
                        # print(rescaling_factor*x+i, rescaling_factor*y+j, w,h)
                        r, g, b = input_image.get(rescaling_factor*x+i, rescaling_factor*y+j)
                        rs.append(r)
                        gs.append(g)
                        bs.append(b)
                pixels = rescaling_factor*rescaling_factor
                new_image.set(x,y, (sum(rs)//pixels, sum(gs)//pixels, sum(bs)//pixels))
                

        return new_image
    else:
        return input_image
    # -----
    
    
def upsample(input_image, rescaling_factor):
    """
        Implement a function that outputs an upsampled version of the input image.

    Args:
        input_image:        image to be upsampled [cs1media Picture object]
        rescaling_factor:   upsampling factor [int]

    Returns:
        upsampled_image:    upsampled image [cs1media Picture object]

    """
    # -----
    # TODO: Write your code here
    w,h = input_image.size()
    newh = h*rescaling_factor
    neww = w*rescaling_factor
    print(neww,newh)
    time.sleep(1)
    new_image = create_picture(neww,newh)
    # add = [[0,0],[0,1],[1,1],[1,0],[-1,0],[-1,-1],[0,-1],[-1,1],[1,-1]]
    for y in range(h):
        for x in range(w):
            orr, org, orb = input_image.get(x, y)
            # for i in add:
            for i in range(-rescaling_factor//2,rescaling_factor//2+1):
                for j in range(-rescaling_factor//2,rescaling_factor//2+1):
                    colorx = i+x*rescaling_factor
                    colory = j+y*rescaling_factor
                    if -1<colorx<neww and -1<colory<newh:
                        new_image.set(colorx,colory, (orr, org, orb))
                        print('좌표:',colorx,colory)

            if x == w-1:
                # new_image.show()
                a=0
                for i in range(-rescaling_factor//2,rescaling_factor//2+1):
                    a+=1
                    endx = x+a
                    if endx < w:
                        new_image.set(endx,i, (orr, org, orb))
                        print('좌표:',endx,i)
            if y == h-1:
                b=0
                for i in range(-rescaling_factor//2,rescaling_factor//2+1):
                    b+=1
                    endy = y+b
                    if endy < h:
                        new_image.set(i,endy, (orr, org, orb))
                        print('좌표:',i,endy)
            if x == w-1 and y == h-1:
                for a in range(rescaling_factor):
                    for b in range(rescaling_factor):
                        new_image.set(x+a,y+b, (orr, org, orb))
                        
                        print('좌표:',x+a,y b)

            # if x != w-1 and y == h-1:
            #     for i in range((rescaling_factor//2)*(rescaling_factor//2+1)):
            #         new_image.set(,, (orr, org, orb))
    return new_image
    # -----


def main():
    # ----------
    # You can try any of the following three images.
    img_path = './images/minion.png'
    # img_path = './images/example.png'
    # img_path = './images/ryan.png'
    # ----------
    image = load_picture(img_path)
    # print(image.size())
    # downsampling_factor = 3
    # downsampled_image = downsample(image, downsampling_factor)
    # downsampled_image.show()
    
    upsampling_factor = 2
    upsampled_image = upsample(image, upsampling_factor)
    upsampled_image.show()


if __name__ == '__main__':
    main()
