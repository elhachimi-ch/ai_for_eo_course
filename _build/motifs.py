import math
def star8(cx,cy,R,ri):
    pts=[]
    for k in range(16):
        r=R if k%2==0 else ri
        a=math.pi/8*k - math.pi/2
        pts.append(f"{cx+r*math.cos(a):.2f},{cy+r*math.sin(a):.2f}")
    return " ".join(pts)
def octagon(cx,cy,R):
    return " ".join(f"{cx+R*math.cos(math.pi/8+math.pi/4*k):.2f},{cy+R*math.sin(math.pi/8+math.pi/4*k):.2f}" for k in range(8))
def tile(S=120, stroke="#C99A2E", sw=1.4, op=1.0, fill="none", fill2="none"):
    R=S*0.30; ri=R*0.70
    parts=[]
    for (x,y) in [(0,0),(S,0),(0,S),(S,S),(S/2,S/2)]:
        parts.append(f'<polygon points="{star8(x,y,R,ri)}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        parts.append(f'<polygon points="{octagon(x,y,R*0.42)}" fill="{fill2}" stroke="{stroke}" stroke-width="{sw*0.8}"/>')
    # connecting kites (4-point stars) at edge midpoints
    for (x,y) in [(S/2,0),(0,S/2),(S,S/2),(S/2,S)]:
        r=S*0.16
        pts=[(x,y-r),(x+r*0.38,y),(x,y+r),(x-r*0.38,y)]
        pts2=[(x-r,y),(x,y-r*0.38),(x+r,y),(x,y+r*0.38)]
        for P in (pts,pts2):
            parts.append('<polygon points="'+" ".join(f"{a:.2f},{b:.2f}" for a,b in P)+f'" fill="none" stroke="{stroke}" stroke-width="{sw*0.8}"/>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{S}" height="{S}" viewBox="0 0 {S} {S}" opacity="{op}">'+"".join(parts)+'</svg>'
def frieze(H=22, W=44, c1="#1F3A93", c2="#C99A2E", c3="#0F6B5C"):
    s=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
    s+=f'<polygon points="{W/2},2 {W/2+9},{H/2} {W/2},{H-2} {W/2-9},{H/2}" fill="{c1}"/>'
    s+=f'<polygon points="{W/2},7 {W/2+4},{H/2} {W/2},{H-7} {W/2-4},{H/2}" fill="{c2}"/>'
    s+=f'<polygon points="0,{H/2} 6,{H/2-5} 12,{H/2} 6,{H/2+5}" fill="{c3}"/><polygon points="{W-12},{H/2} {W-6},{H/2-5} {W},{H/2} {W-6},{H/2+5}" fill="{c3}"/>'
    s+=f'<line x1="0" y1="1" x2="{W}" y2="1" stroke="{c2}" stroke-width="1.2"/><line x1="0" y1="{H-1}" x2="{W}" y2="{H-1}" stroke="{c2}" stroke-width="1.2"/>'
    return s+'</svg>'
def rosette(size=200, c="#C99A2E", c2="#1F3A93", c3="#0F6B5C"):
    cx=cy=size/2
    s=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}">'
    for R,ri,col,fill in [(size*0.48,size*0.33,c,'none'),(size*0.40,size*0.28,c2,c2),(size*0.26,size*0.18,c,c),(size*0.15,size*0.10,c3,c3)]:
        s+=f'<polygon points="{star8(cx,cy,R,ri)}" fill="{fill}" stroke="{col}" stroke-width="2"/>'
    s+=f'<circle cx="{cx}" cy="{cy}" r="{size*0.05}" fill="#FBF7F0"/>'
    return s+'</svg>'
if __name__=="__main__":
    import sys
    open('tile_preview.svg','w').write(tile())
    open('frieze_preview.svg','w').write(frieze())
    open('rosette_preview.svg','w').write(rosette())
