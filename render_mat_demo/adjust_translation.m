function [ sim ] = adjust_translation( yaw )
tx=0;
ty=0;
switch yaw
    case 0
        tx=-20;
        ty=0;
    case -22
        tx=-30;
        ty=0;
    case -40
        tx=-32;
        ty=0;
     case -55
        tx=-21;
        ty=0;
    case -70
        tx=-21;
        ty=0;
    case -75
        tx=-21;
        ty=0;
end
sim.tx = tx;
sim.ty = ty;
end