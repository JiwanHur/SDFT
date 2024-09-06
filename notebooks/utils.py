import PIL.Image as Image
import torchvision
to_pil = torchvision.transforms.ToPILImage()

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def concat_one_batch(batch_size, batch1):
    out = None
    for batch_idx in range(batch_size):
        if out is not None:
            out = get_concat_v(
                out, to_pil(batch1[batch_idx])
            )
        else:
            out = to_pil(batch1[batch_idx])
    return out

def concat_two_batch(batch_size, batch1, batch2):
    out = None
    for batch_idx in range(batch_size):
        if out is not None:
            out = get_concat_v(
                out, get_concat_h(to_pil(batch1[batch_idx]), to_pil(batch2[batch_idx]))
            )
        else:
            out = get_concat_h(to_pil(batch1[batch_idx]), to_pil(batch2[batch_idx]))
    return out

def concat_three_batch(batch_size, batch1, batch2, batch3):
    out = None
    for batch_idx in range(batch_size):
        if out is not None:
            out = get_concat_v(
                out, get_concat_h(
                get_concat_h(to_pil(batch1[batch_idx]), to_pil(batch2[batch_idx])), to_pil(batch3[batch_idx])
                )
            )
        else:
            out = get_concat_h(
                get_concat_h(to_pil(batch1[batch_idx]), to_pil(batch2[batch_idx])), to_pil(batch3[batch_idx])
            )
    return out

def concat_four_batch(batch_size, batch1, batch2, batch3, batch4):
    out = None
    for batch_idx in range(batch_size):
        if out is not None:
            out = get_concat_v(
                out, get_concat_h(get_concat_h(get_concat_h(
                    to_pil(batch1[batch_idx]), to_pil(batch2[batch_idx])), to_pil(batch3[batch_idx])), to_pil(batch4[batch_idx])
                )
            )
        else:
            out = get_concat_h(get_concat_h(get_concat_h(
                to_pil(batch1[batch_idx]), to_pil(batch2[batch_idx])), to_pil(batch3[batch_idx])), to_pil(batch4[batch_idx])
            )
    return out

def concat_five_batch(batch_size, batch1, batch2, batch3, batch4, batch5):
    out = None
    for batch_idx in range(batch_size):
        if out is not None:
            out = get_concat_v(
                out, get_concat_h(get_concat_h(get_concat_h(get_concat_h(
                    to_pil(batch1[batch_idx]), to_pil(batch2[batch_idx])), to_pil(batch3[batch_idx])), to_pil(batch4[batch_idx])),to_pil(batch5[batch_idx])
                )
            )
        else:
            out = get_concat_h(get_concat_h(get_concat_h(get_concat_h(
                to_pil(batch1[batch_idx]), to_pil(batch2[batch_idx])), to_pil(batch3[batch_idx])), to_pil(batch4[batch_idx])),to_pil(batch5[batch_idx])
            )
    return out

def concat_six_batch(batch_size, batch1, batch2, batch3, batch4, batch5, batch6):
    out = None
    for batch_idx in range(batch_size):
        if out is not None:
            out = get_concat_v(
                out, get_concat_h(get_concat_h(get_concat_h(get_concat_h(get_concat_h(
                    to_pil(batch1[batch_idx]), to_pil(batch2[batch_idx])), to_pil(batch3[batch_idx])), to_pil(batch4[batch_idx])),to_pil(batch5[batch_idx])) ,to_pil(batch6[batch_idx])
                )
            )
        else:
            out = get_concat_h(get_concat_h(get_concat_h(get_concat_h(get_concat_h(
                to_pil(batch1[batch_idx]), to_pil(batch2[batch_idx])), to_pil(batch3[batch_idx])), to_pil(batch4[batch_idx])),to_pil(batch5[batch_idx])) ,to_pil(batch6[batch_idx])
            )
    return out

def concat_seven_batch(batch_size, batch1, batch2, batch3, batch4, batch5, batch6, batch7):
    out = None
    for batch_idx in range(batch_size):
        if out is not None:
            out = get_concat_v(
                out, get_concat_h(get_concat_h(get_concat_h(get_concat_h(get_concat_h(get_concat_h(
                    to_pil(batch1[batch_idx]), to_pil(batch2[batch_idx])), to_pil(batch3[batch_idx])), to_pil(batch4[batch_idx])),to_pil(batch5[batch_idx])) ,to_pil(batch6[batch_idx])) ,to_pil(batch7[batch_idx])
                )
            )
        else:
            out = get_concat_h(get_concat_h(get_concat_h(get_concat_h(get_concat_h(get_concat_h(
                to_pil(batch1[batch_idx]), to_pil(batch2[batch_idx])), to_pil(batch3[batch_idx])), to_pil(batch4[batch_idx])),to_pil(batch5[batch_idx])) ,to_pil(batch6[batch_idx])) ,to_pil(batch7[batch_idx])
            )
    return out

def concat_eight_batch(batch_size, batch1, batch2, batch3, batch4, batch5, batch6, batch7, batch8):
    out = None
    for batch_idx in range(batch_size):
        if out is not None:
            out = get_concat_v(
                out, get_concat_h(get_concat_h(get_concat_h(get_concat_h(get_concat_h(get_concat_h(get_concat_h(
                    to_pil(batch1[batch_idx]), to_pil(batch2[batch_idx])), to_pil(batch3[batch_idx])), to_pil(batch4[batch_idx])),to_pil(batch5[batch_idx])) ,to_pil(batch6[batch_idx])) ,to_pil(batch7[batch_idx])),to_pil(batch8[batch_idx])
                )
            )
        else:
            out = get_concat_h(get_concat_h(get_concat_h(get_concat_h(get_concat_h(get_concat_h(get_concat_h(
                to_pil(batch1[batch_idx]), to_pil(batch2[batch_idx])), to_pil(batch3[batch_idx])), to_pil(batch4[batch_idx])),to_pil(batch5[batch_idx])) ,to_pil(batch6[batch_idx])) ,to_pil(batch7[batch_idx])) ,to_pil(batch8[batch_idx])
            )
    return out