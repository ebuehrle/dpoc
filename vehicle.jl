function vehicle(px,py,vx,vy,l,w)
    x = [px, py]
    v = [vx, vy]
    vf = v / sqrt(v'v)
    vl = [-vf[2], vf[1]]
    vv = [
        (x + vf*l/2)';
        (x + vf*l/2 + vl*w/2)';
        (x - vf*l/2 + vl*w/2)';
        (x - vf*l/2 - vl*w/2)';
        (x + vf*l/2 - vl*w/2)';
        (x + vf*l/2)';
        (x + vl*w/2)';
        (x - vl*w/2)';
        (x + vf*l/2)';
    ]'
    return vv
end
