function [Nv,dNdxi]=lagrange_basis(type,coord,dim)
% =========================================================================
% Returns the lagrange interpolant basis and its gradients w.r.t the
% parent coordinate system.
% =========================================================================

switch type
    case 'L2'
        %%%%%%%%%%%%%%%%%%%%% L2 TWO NODE LINE ELEMENT %%%%%%%%%%%%%%%%%%%%%
        %
        %    1---------2
        %
        if size(coord,2) < 1
            disp('Error coordinate needed for the L2 element')
        else
            xi=coord(1);
            N=([1-xi,1+xi]/2)';
            dNdxi=[-1;1]/2;
        end
        
    case 'L3'
        %%%%%%%%%%%%%%%%%%% L3 THREE NODE LINE ELEMENT %%%%%%%%%%%%%%%%%%%%%
        %
        %    1---------2----------3
        %
        if size(coord,2) < 1
            disp('Error two coordinates needed for the L3 element')
        else
            xi=coord(1);
            N=[(1-xi)*xi/(-2);(1+xi)*xi/2;1-xi^2];
            dNdxi=[xi-.5;xi+.5;-2*xi];
        end
        
    case 'T3'
        %%%%%%%%%%%%%%%% T3 THREE NODE TRIANGULAR ELEMENT %%%%%%%%%%%%%%%%%%
        %
        %               3
        %             /  \
        %            /    \
        %           /      \
        %          /        \
        %         /          \
        %        /            \
        %       /              \
        %      /                \
        %     /                  \
        %    1--------------------2
        %
        if size(coord,2) < 2
            disp('Error two coordinates needed for the T3 element')
        else
            xi=coord(1); eta=coord(2);
            N=[1-xi-eta;xi;eta];
            dNdxi=[-1,-1;1,0;0,1];
        end
        
    case 'T4'
        %%%%%%%%%% T4 FOUR NODE TRIANGULAR CUBIC BUBBLE ELEMENT %%%%%%%%%%%%
        %
        %               3
        %             /  \
        %            /    \
        %           /      \
        %          /        \
        %         /          \
        %        /      4     \
        %       /              \
        %      /                \
        %     /                  \
        %    1--------------------2
        %
        if size(coord,2) < 2
            disp('Error two coordinates needed for the T4 element')
        else
            xi=coord(1); eta=coord(2);
            N=[1-xi-eta-3*xi*eta;xi*(1-3*eta);eta*(1-3*xi);9*xi*eta];
            dNdxi=[-1-3*eta,-1-3*xi;
                1-3*eta, -3*xi;
                -3*eta,   1-3*xi;
                9*eta,   9*xi ];
        end
        
    case 'T6'
        %%%%%%%%%%%%%%%%%% T6 SIX NODE TRIANGULAR ELEMENT %%%%%%%%%%%%%%%%%%
        %
        %               3
        %             /  \
        %            /    \
        %           /      \
        %          /        \
        %         6          5
        %        /            \
        %       /              \
        %      /                \
        %     /                  \
        %    1---------4----------2
        %
        if size(coord,2) < 2
            disp('Error two coordinates needed for the T6 element')
        else
            xi=coord(1); eta=coord(2);
            N=[1-3*(xi+eta)+4*xi*eta+2*(xi^2+eta^2);
                xi*(2*xi-1);
                eta*(2*eta-1);
                4*xi*(1-xi-eta);
                4*xi*eta;
                4*eta*(1-xi-eta)];
            
            dNdxi=[4*(xi+eta)-3   4*(xi+eta)-3;
                4*xi-1              0;
                0        4*eta-1;
                4*(1-eta-2*xi)          -4*xi;
                4*eta           4*xi;
                -4*eta  4*(1-xi-2*eta)];
        end
        
        
    case 'Q4'
        %%%%%%%%%%%%%%% Q4 FOUR NODE QUADRILATERIAL ELEMENT %%%%%%%%%%%%%%%%
        %
        %    4--------------------3
        %    |                    |
        %    |                    |
        %    |                    |
        %    |                    |
        %    |                    |
        %    |                    |
        %    |                    |
        %    |                    |
        %    |                    |
        %    1--------------------2
        %
        if size(coord,2) < 2
            disp('Error two coordinates needed for the Q4 element')
        else
            xi=coord(1); eta=coord(2);
            N=1/4*[ (1-xi)*(1-eta);
                (1+xi)*(1-eta);
                (1+xi)*(1+eta);
                (1-xi)*(1+eta)];
            dNdxi=1/4*[-(1-eta), -(1-xi);
                1-eta,    -(1+xi);
                1+eta,      1+xi;
                -(1+eta),   1-xi];
        end
        
    case 'Q9'
        %%%%%%%%%%%%%%% Q9 NINE NODE QUADRILATERIAL ELEMENT %%%%%%%%%%%%%%%%
        %
        %    4---------7----------3
        %    |                    |
        %    |                    |
        %    |                    |
        %    |                    |
        %    8          9         6
        %    |                    |
        %    |                    |
        %    |                    |
        %    |                    |
        %    1----------5---------2
        %
        if size(coord,2) < 2
            disp('Error two coordinates needed for the Q9 element')
        else
            xi=coord(1); eta=coord(2);
            N=1/4*[xi*eta*(xi-1)*(eta-1);
                xi*eta*(xi+1)*(eta-1);
                xi*eta*(xi+1)*(eta+1);
                xi*eta*(xi-1)*(eta+1);
                -2*eta*(xi+1)*(xi-1)*(eta-1);
                -2*xi*(xi+1)*(eta+1)*(eta-1);
                -2*eta*(xi+1)*(xi-1)*(eta+1);
                -2*xi*(xi-1)*(eta+1)*(eta-1);
                4*(xi+1)*(xi-1)*(eta+1)*(eta-1)];
            dNdxi=1/4*[eta*(2*xi-1)*(eta-1),xi*(xi-1)*(2*eta-1);
                eta*(2*xi+1)*(eta-1),xi*(xi+1)*(2*eta-1);
                eta*(2*xi+1)*(eta+1),xi*(xi+1)*(2*eta+1);
                eta*(2*xi-1)*(eta+1),xi*(xi-1)*(2*eta+1);
                -4*xi*eta*(eta-1),   -2*(xi+1)*(xi-1)*(2*eta-1);
                -2*(2*xi+1)*(eta+1)*(eta-1),-4*xi*eta*(xi+1);
                -4*xi*eta*(eta+1),   -2*(xi+1)*(xi-1)*(2*eta+1);
                -2*(2*xi-1)*(eta+1)*(eta-1),-4*xi*eta*(xi-1);
                8*xi*(eta^2-1),      8*eta*(xi^2-1)];
        end
        
    case 'H4'
        %%%%%%%%%%%%%%%% H4 FOUR NODE TETRAHEDRAL ELEMENT %%%%%%%%%%%%%%%%%%
        %
        %             4
        %           / | \
        %          /  |  \
        %         /   |   \
        %        /    |    \
        %       /     |     \
        %      1 -----|------2
        %         -   3  -
        if size(coord,2) < 3
            disp('Error three coordinates needed for the H4 element')
        else
            xi=coord(1); eta=coord(2); zeta=coord(3);
            N=[1-xi-eta-zeta;
                xi;
                eta;
                zeta];
            dNdxi=[-1  -1  -1;
                1   0   0;
                0   1   0;
                0   0   1];
        end
        
        
    case 'B8'
        %%%%%%%%%%%%%%%%%%% B8 EIGHT NODE BRICK ELEMENT %%%%%%%%%%%%%%%%%%%%
        %
        %                  8
        %               /    \
        %            /          \
        %         /                \
        %      5                     \
        %      |\                     7
        %      |   \                / |
        %      |     \     4    /     |
        %      |        \    /        |
        %      |           6          |
        %      1           |          |
        %       \          |          3
        %          \       |        /
        %            \     |     /
        %               \  |  /
        %                  2
        %
        if size(coord,2) < 3
            disp('Error three coordinates needed for the B8 element')
        else
            xi=coord(1); eta=coord(2); zeta=coord(3);
            
            N = [1/8*(1-xi)*(1-eta)*(1-zeta);
                1/8*(1+xi)*(1-eta)*(1-zeta);
                1/8*(1+xi)*(1+eta)*(1-zeta);
                1/8*(1-xi)*(1+eta)*(1-zeta);
                1/8*(1-xi)*(1-eta)*(1+zeta);
                1/8*(1+xi)*(1-eta)*(1+zeta);
                1/8*(1+xi)*(1+eta)*(1+zeta);
                1/8*(1-xi)*(1+eta)*(1+zeta)];
            
            
            dNdxi=[   -1+eta+zeta-eta*zeta   -1+xi+zeta-xi*zeta  -1+xi+eta-xi*eta;
                1-eta-zeta+eta*zeta   -1-xi+zeta+xi*zeta  -1-xi+eta+xi*eta;
                1+eta-zeta-eta*zeta    1+xi-zeta-xi*zeta  -1-xi-eta-xi*eta;
                -1-eta+zeta+eta*zeta    1-xi-zeta+xi*zeta  -1+xi-eta+xi*eta;
                -1+eta-zeta+eta*zeta   -1+xi-zeta+xi*zeta   1-xi-eta+xi*eta;
                1-eta+zeta-eta*zeta   -1-xi-zeta-xi*zeta   1+xi-eta-xi*eta;
                1+eta+zeta+eta*zeta    1+xi+zeta+xi*zeta   1+xi+eta+xi*eta;
                -1-eta-zeta-eta*zeta    1-xi+zeta-xi*zeta   1-xi+eta-xi*eta ]/8;
            
        end
        
    otherwise
        disp(['Element ',type,' not yet supported'])
        N=[]; dNdxi=[];
end

if ( nargin == 2 )
    Nv = N;
else
    Nv=[];
    for i=1:size(N,1)
        Nv=[Nv;eye(dim)*N(i)];
    end
end

