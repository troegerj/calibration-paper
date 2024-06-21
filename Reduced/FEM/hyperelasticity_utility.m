function [cauchy,ct] = hyperelasticity_utility(detF,Bq,K,w1,w2,w11,w12,w22)
% DESCRIPTION
%  Function providing Cauchy stress and consistent tangent operator for 
%  finite strain hyperelastic materials. All computations are carried out
%  with respect to the current configuration and are based on the
%  volumetric/isochoric split of the strain energy density.
%
%  Ansatz for volumetric part of strain energy density:
%   U(J) = K/2 * (J-1)^2 with J = det F
%
%  Ansatz for isochoric part is denoted with w's (partial derivatives of 
%  isochoric strain energy density w.r.t. first and second invariant of 
%  unimodular left Cauchy-Green tensor) as input. 
%
%  Specific derivations of the stress tensors and consistent tangent
%  operator are provided by
%   S. Hartmann - Finite-Elemente Berechnung inelastischer Kontinua, 2003,
%   pg. 76 onwards
%
% INPUT 
%  detF -- determinant of deformation gradient
%  Bq -- unimodular left Cauchy-Green tensor (stored in compact notation)
%   Bq = [Bq11 Bq22 Bq33 Bq12 Bq23 Bq31]
%  K -- bulk modulus [N/mm^2]
%  w1 -- see above, dw/dIBq
%  w2 -- see above, dw/dIIBq
%  w11 -- see above, d^2w/dIBq^2
%  w12 -- see above, d^2w/(dIBq dIIBq)
%  w22 -- see above, d^2w/dIIBq^2
%
% OUTPUT 
%  cauchy -- Cauchy stress tensor (stored in compact notation)
%  ct -- consistent tangent operator (stored in compact notation)

% evaluate volumetric part of strain energy density
% U(1) is U(J) = K/2 * (J-1)^2 with J = det F
% U(2) is U'(J) = K(J-1)
% U(3) is U''(J) = K
U(1) = 0.5*K*(detF-1)^2;
U(2) = K*(detF-1);
U(3) = K;

%% stress computation

% compute product Bq Bq
Bq2 = prodAB(Bq,Bq);

% compute invariants of unimodular left Cauchy-Green tensor
IBq = Bq(1)+Bq(2)+Bq(3);
IIBq = 0.5*(IBq^2-(Bq2(1)+Bq2(2)+Bq2(3)));

% compute isochoric part of Kirchhoff stress tensor
Siso = 2*(w1 + w2*IBq)*Bq - 2*w2*Bq2;
% compute deviatoric part of isochoric Kirchhoff stress tensor
trSiso = Siso(1)+Siso(2)+Siso(3); % tr(S_iso)
SisoD(1:3) = Siso(1:3) - trSiso/3;
SisoD(4:6) = Siso(4:6);

% compute volumetrical/spherical part of Kirchhoff stress tensor
Svol(1:3) = detF*U(2);  % J*U'(J)
Svol(4:6) = 0.0;

% add volumetrical and isochoric part of Kirchhoff stress tensor
S = Svol + SisoD;

% compute Cauchy stress tensor
cauchy = 1/detF * S;

%% consistent tangent

% compute volumetric part of consistent tangent operator
Cvol = zeros(6,6);
for i = 1 : 3
    for j = 1 : 3
        Cvol(i,j) = U(2) + detF*U(3); % U'(J) + J*U''(J)
    end
    Cvol(i,i) = Cvol(i,i) - 2*U(2);
    Cvol(i+3,i+3) = -2*U(2);
end

% calculate isochoric part of consistent tangent operator
%
% compute -w2*(Bq o Bq)^T23
mat1 = -w2*bobt23(Bq);
% compute (w11 + 2w12*IBq + w22*IBq^2 + w2)*(Bq o Bq)
mat2 = (w11 + 2*w12*IBq + w22*IBq^2 + w2)*tenmat(Bq,Bq);
% compute -(w12 + IBq*w22)*(Bq^2 o Bq) + (Bq o Bq^2)
mat3 = -(w12 + IBq*w22)*abba(Bq,Bq2);
% compute w22*(Bq^2 o Bq^2)
mat4 = w22*tenmat(Bq2,Bq2);
% sum mat1...mat4 to get 4(Bq o I)^T23 d^2w/dBq^2 (I o Bq)^T23
mat1 = 4*(mat1 + mat2 + mat3 + mat4);
clear mat2 mat3 mat4
% pre- and post-multiply with fourth-order deviator operator
mat1 = devMult(mat1);

% compute further part of consistent tangent operator
%  4/3 * (dw/dBq \cdot Bq) Dev 
mat2 = (2/3) * trSiso;
mat2 = devFac(mat2);

% compute final part of consistent tangent operator
%  -2/3 * (Siso o I + I o Siso)
one = [1 1 1 0 0 0]'; % second-order identity tensor in compact notation
mat3 = -(2/3) * abba(SisoD,one);

% compute isochoric part of consistent tangent operator
Ciso = 1/detF * (mat1 + mat2 + mat3);

% only the upper right triangular part of the tangent operator is computed,
% thus symmetrization step and factorization is required
Cvol = symmet(Cvol);
Ciso = symmet(Ciso);
Cvol(:,4:6) = 0.5*Cvol(:,4:6);
Ciso(:,4:6) = 0.5*Ciso(:,4:6);

% compute consistent tangent operator
ct = Cvol + Ciso;

end

%% subfunctions

function C = prodAB(A,B)
% function computing tensor product C = AB for two symmetric second-order
% tensors A and B, each stored in column form A = [a11 a22 a33 a12 a23
% a31]. Output is 6x1 array (again using compact notation).

C = zeros(6);

C(1) = A(1)*B(1) + A(4)*B(4) + A(6)*B(6);
C(2) = A(4)*B(4) + A(2)*B(2) + A(5)*B(5);
C(3) = A(6)*B(6) + A(5)*B(5) + A(3)*B(3);
%
C(4) = A(1)*B(4) + A(4)*B(2) + A(6)*B(5);
C(5) = A(4)*B(6) + A(2)*B(5) + A(5)*B(3);
C(6) = A(6)*B(1) + A(5)*B(4) + A(3)*B(6);

end


function C = bobt23(B)
% function computing C = (B o B)^T23 for symmetric second-order tensor B, 
% stored in column form B = [b11 b22 b33 b12 b23 b31]. Output is 6x6 matrix
% containing the reduced tensor components of the product.

% initialize
C = zeros(6,6);

for i = 1 : 3
    C(i,i) = B(i)*B(i);
end
%
C(1,2) = B(4)*B(4);
C(1,3) = B(6)*B(6);
C(2,3) = B(5)*B(5);
%
C(1,4) = 2*B(1)*B(4);
C(1,5) = 2*B(4)*B(6);
C(1,6) = 2*B(1)*B(6);
%
C(2,4) = 2*B(2)*B(4);
C(2,5) = 2*B(2)*B(5);
C(2,6) = 2*B(4)*B(5);
%
C(3,4) = 2*B(5)*B(6);
C(3,5) = 2*B(3)*B(5);
C(3,6) = 2*B(3)*B(6);
%
C(4,4) = B(1)*B(2) + C(1,2);
C(4,5) = B(2)*B(6) + B(4)*B(5);
C(4,6) = B(4)*B(6) + B(1)*B(5);
%
C(5,5) = B(2)*B(3) + C(2,3);
C(5,6) = B(3)*B(4) + B(5)*B(6);
%
C(6,6) = B(1)*B(3) + C(1,3);

end


function C = tenmat(A,B)
% function returns the dyadic product of two symmetric second-order tensors 
% A and B, stored in column format (A = [a11 a22 a33 a12 a23 a31]). Output 
% is 6x6 matrix containing the fourth-order tensor components of the product.

% evaluate dyadic product
C = A*B';
% consider symmetry of tensors
C(4:6,4:6) = 2*C(4:6,4:6);

end


function C = abba(A,B) 
% function provides the sum of the two dyadic tensor products 
% C = (A o B) + (B o A) for two symmetric second-order tensors A and B
% stored in column form.

% initialize
C = zeros(6,6);

for i = 1 : 3
    C(i,i) = 2*A(i)*B(i);
    C(i+3,i+3) = 4*A(i+3)*B(i+3);
end
%
for i = 1 : 6
    for j = 1 : 6
        if i > j
            continue;
        else
            if j <= 3
                C(i,j) = A(i)*B(j) + A(j)*B(i);
            elseif j > 3
                C(i,j) = 2*(A(i)*B(j) + A(j)*B(i));
            end
        end
    end
end

end


function B = devMult(A)
% tensor product B = D A D with fourth-order tensor A (stored in compact 
% notation due to symmetry) and D = 1 - (1/3) (I o I) as fourth-order
% deviator operator with second-order identity tensor I and fourth order
% identity tensor 1.

% initialize
B = zeros(6,6);

% define upper left block matrix of deviator operator
dev = (1/3)*[2 -1 -1; -1 2 -1; -1 -1 2];

A(2,1) = A(1,2);
A(3,1) = A(1,3); 
A(3,2) = A(2,3);
for i = 1 : 3
    for j = i : 3
        for k = 1 : 3
            for l = 1 : 3
                B(i,j) = B(i,j) + dev(i,k)*A(k,l)*dev(l,j);
            end
        end
    end
end
               
for i = 1 : 3
    for j = 4 : 6
        for k = 1 : 3
            B(i,j) = B(i,j) + dev(i,k)*A(k,j);
        end
    end
end

A(5,4) = A(4,5);
A(6,4) = A(4,6);
A(6,5) = A(5,6);
for i = 4 : 6
    for j = i : 6
        B(i,j) = A(i,j);
    end
end

end


function A = devFac(alpha)
% multiplication of fourth-order deviator operator D = 1 - (1/3) (I o I)
% with scalar factor alpha: A = alpha*D

% define deviator operator
dev = zeros(6,6);
dev(1:3,1:3) = (1/3)*[2 -1 -1; 0 2 -1; 0 0 2];
dev(4:6,4:6) = eye(3,3);

% consider scalar factor
A = alpha*dev;

end


function A = symmet(A)
% symmetrization of a given 6x6-matrix (upper left triangular matrix). It 
% must be emphasized that the columns 4-6 of matrix A are twice as big as
% the rows 4-6

for i = 2 : 3 
    ii = i + 3;
    for j = 1 : i-1
        jj       = j + 3;
        A(i,j)   = A(j,i);
        A(ii,jj) = A(jj,ii);
    end
end
%
for i = 4 : 6
    for j = 1 : 3
        A(i,j) = 0.5*A(j,i);
    end
end

end