function [y, Lines] = AsymRotor(x, A, B, C, dA, dB, dC, Nu0, Trot, Theta, Phi, GW, LW, JMax, Unit, SpinAxis, SpinOdd, T2, W2)
%ASYMROTOR  Rotationally resolved band of an asymmetric top (rigid rotor).
%
%   y = ASYMROTOR(x, A, B, C, dA, dB, dC, Nu0, Trot, Theta, Phi, GW, LW)
%   returns the band of a transition between two vibronic states of an
%   asymmetric-top molecule, on the axis x, with unit area.  It is a model
%   component, typed into the equation box, e.g. with an Offset (C1):
%
%       RInt1*AsymRotor(x-923500000, RA1, RB1, RC1, RdA1, RdB1, RdC1, Nu01, ...
%                       Trot1, Th1, Ph1, GW1, LW1)
%
%   RInt1 is then the integrated intensity of the whole band, in the units
%   of the data times the units of x.  
%
%   x          frequency axis, in the units of the constants (MHz by default)
%   A, B, C    rotational constants of the LOWER state
%   dA,dB,dC   upper minus lower: A' = A + dA, and so on.  Fit these rather
%              than A', B', C': line positions depend on the sums and on the
%              small differences, so A' and A'' fitted separately are
%              strongly correlated and dA, dB, dC are not.
%   Nu0        band origin, in the units of x.  It IS the calibration shift:
%              a constant offset of the axis and the origin move every line
%              together, so the two cannot be fitted at once.
%   Trot       rotational temperature, K
%   Theta, Phi direction of the transition moment, degrees: Theta from the a
%              axis, Phi about it from b towards c.  mu_a^2 = cos^2(Theta),
%              mu_b^2 = sin^2(Theta) cos^2(Phi), mu_c^2 = sin^2(Theta)
%              sin^2(Phi).  a-type 0/0, b-type 90/0, c-type 90/90.  Only
%              the squares enter (no axis switching), so the signs do not.
%   GW, LW     Gaussian (Doppler) and Lorentzian (lifetime) FWHM, units of x.
%              The profile is an exact Voigt.
%
%   Optional, in this order (all numeric, so the call can sit in a model
%   equation fittype reads; [] or an omitted argument takes the default):
%
%   JMax       highest J of the lower state.  0 (default) picks it from Trot
%              and C, rounded up to a multiple of 10.  Fix it when Trot or T2
%              is free, so that moving the temperature never changes the set
%              of levels (and the cache below keeps working).
%   Unit       units of x and of the constants, for the Boltzmann factor:
%              1 MHz (default), 2 GHz, 3 cm-1
%   SpinAxis   the C2 axis that exchanges identical nuclei: 0 none (default),
%              1 a, 2 b, 3 c
%   SpinOdd    statistical weight of the lower levels that are odd under that
%              C2 rotation, relative to the even ones (default 1).  For
%              carbazole (C2 along b, four pairs of protons exchanged) it is
%              120/136 = 15/17 with an A1 ground state.
%   T2, W2     second temperature and its weight (the two-temperature model
%              of a supersonic expansion): populations go as
%              exp(-E/kTrot) + W2*exp(-E/kT2).  W2 = 0 (default) is off.
%
%   [y, Lines] = ASYMROTOR(...) also returns the stick list (Nu, I, Type =
%   1/2/3 for a/b/c, J/Ka/Kc of both levels, the lower level's energy),
%   sorted by frequency, every line above 1e-6 of the strongest - the
%   assignment of what the fit is matching.
%
%   The physics
%   -----------
%   Each J block of H = A Ja^2 + B Jb^2 + C Jc^2 is diagonalised in the
%   symmetric-top basis |J,K> (representation I^r: z = a, x = b, y = c),
%   split into the four Wang blocks (K even/odd, symmetric/antisymmetric in
%   K <-> -K) so that every eigenvector has an exact symmetry even where the
%   K doublets are degenerate to machine precision.  The line strength of
%   |J' t'> <- |J t> is (2J+1) |sum_K c'(K+q) c(K) <J K; 1 q|J' K+q>|^2,
%   with q = 0 for a-type and (q = -1) -+ (q = +1) over sqrt(2) for b and c,
%   which reduces to the Honl-London factors in the symmetric-top limit and
%   sums to 2J+1 over all upper levels for every lower level.  Intensity is
%   line strength x mu^2 x spin weight x Boltzmann factor of the lower level.
%   Centrifugal distortion is not included (rigid rotor).
%
%   The levels and line strengths depend on A, B, C, dA, dB, dC and JMax
%   only; they are kept in a small cache, so a fit that moves Nu0, Trot,
%   Theta, Phi, GW, LW or the intensity re-diagonalises nothing.  On a grid
%   over the rotational constants each grid point pays for one build, and
%   the fit at that point is cheap.
%
%   Positions are exact; the broadening is done on a fine grid (40 points
%   per FWHM, nodes at integer multiples of its step so that a subset of x
%   gives the same numbers) by FFT, then interpolated onto x: within 4e-4
%   of the peak of an exact Voigt.  x may be any subset of the data, in any
%   order, and need not be evenly spaced.


y = zeros(size(x));
Lines = [];
% fittype test-evaluates the model with dummy values before any fit
% (FitTypeProbeGuard): rotational constants of 1 would ask for thousands
% of J blocks to compute a curve that is thrown away.
if isequal(getappdata(0,'FitTypeProbe'), true)
    return;
end
if isempty(x)
    return;
end

if nargin < 14 || isempty(JMax),     JMax = 0;     end
if nargin < 15 || isempty(Unit),     Unit = 1;     end
if nargin < 16 || isempty(SpinAxis), SpinAxis = 0; end
if nargin < 17 || isempty(SpinOdd),  SpinOdd = 1;  end
if nargin < 18 || isempty(T2),       T2 = 0;       end
if nargin < 19 || isempty(W2),       W2 = 0;       end

switch round(Unit)
    case 2,    kB = 20.83661912;      % GHz per K
    case 3,    kB = 0.69503476;       % cm-1 per K
    otherwise, kB = 20836.61912;      % MHz per K
end

Up = [A + dA, B + dB, C + dC]; % Up = S1 surface
if any(~isfinite([A B C Up Nu0 Trot GW LW])) || any([A B C Up] <= 0) || Trot <= 0
    error('AsymRotor:BadInput', ...
        'Rotational constants (lower and upper) and Trot must be finite and positive.');
end

% Highest lower-state J: below exp(-14) of the population of J = 0 for the
% lowest level of that J (which lies at or above C*J(J+1)), rounded up to a
% multiple of 10 so that a temperature moving in a fit rarely changes it.
if JMax <= 0
    kTm = kB*Trot;
    if W2 > 0 && T2 > 0
        kTm = max(kTm, kB*T2);
    end
    JMax = 10*ceil(sqrt(14*kTm/min([C, Up(3)]))/10);
end
JMax = min(round(JMax), 400);

Lst = AsymRotorLines(A, B, C, Up(1), Up(2), Up(3), JMax);

% --- intensities --------------------------------------------------------
mu2 = [cosd(Theta)^2, (sind(Theta)*cosd(Phi))^2, (sind(Theta)*sind(Phi))^2];
I = mu2(Lst.Type(:)).' .* Lst.S;
Pop = exp(-Lst.Elow/(kB*Trot));
if W2 > 0 && T2 > 0
    Pop = Pop + W2*exp(-Lst.Elow/(kB*T2));
end
I = I .* Pop;
switch round(SpinAxis)
    case 1, Odd = Lst.Odd(:,1);
    case 2, Odd = Lst.Odd(:,2);
    case 3, Odd = Lst.Odd(:,3);
    otherwise, Odd = false(size(I));
end
I(Odd) = I(Odd)*SpinOdd;
Tot = sum(I);
if ~(Tot > 0)
    return;
end
I = I/Tot;
Nu = Nu0 + Lst.Nu;

if nargout > 1
    keep = I > 1e-6*max(I);
    [~, o] = sort(Nu(keep));
    f = find(keep); f = f(o);
    Lines = struct('Nu', Nu(f), 'I', I(f), 'Type', Lst.Type(f), ...
        'Jl', Lst.Lab(f,1), 'Kal', Lst.Lab(f,2), 'Kcl', Lst.Lab(f,3), ...
        'Ju', Lst.Lab(f,4), 'Kau', Lst.Lab(f,5), 'Kcu', Lst.Lab(f,6), ...
        'Elow', Lst.Elow(f));
end

% --- broadening ---------------------------------------------------------
GW = abs(GW); LW = abs(LW);
fV = 0.5346*LW + sqrt(0.2166*LW^2 + GW^2);   % Voigt FWHM (Olivero-Longbothum)
xs = sort(x(:));
if ~(fV > 0)
    dx = diff(xs); dx = dx(dx > 0);
    if isempty(dx), return; end
    fV = min(dx);                              % no width given: one step
    GW = fV;
end
Pad = 60*LW + 6*GW + 2*fV;                     % Lorentzian tails included
% Independent stretches of x (a merged vector, or a wholly excluded block)
% are broadened separately, so a large gap costs no grid points.
Brk = [0; find(diff(xs) > 4*Pad); numel(xs)];
xc = x(:);
yc = zeros(size(xc));
for s = 1:numel(Brk)-1
    lo = xs(Brk(s)+1) - Pad;
    hi = xs(Brk(s+1)) + Pad;
    in = Nu >= lo & Nu <= hi;
    if ~any(in)
        continue;
    end
    d = fV/40;
    if (hi - lo)/d > 2^22
        d = (hi - lo)/2^22;
    end
    i0 = floor(lo/d);
    i1 = ceil(hi/d);
    N = i1 - i0 + 1;
    p = Nu(in)/d - i0;                         % 0-based node position
    k = floor(p);
    t = p - k;
    Sk = accumarray([k + 1; k + 2], [I(in).*(1 - t); I(in).*t], [N + 1, 1]);
    Nf = 2^nextpow2(N + 1);
    fq = [0:Nf/2, -(Nf/2 - 1):-1].'/(Nf*d);
    Kv = exp(-pi*LW*abs(fq) - (pi*GW*fq).^2/(4*log(2)));
    g = real(ifft(fft(Sk, Nf).*Kv))/d;
    xg = (i0:i1 + 1).'*d;
    sel = xc >= lo & xc <= hi;
    yc(sel) = interp1(xg, g(1:N + 1), xc(sel), 'spline');
end
y(:) = yc;

end


% -------------------------------------------------------------------------
function Lst = AsymRotorLines(A, B, C, Au, Bu, Cu, JMax)
%ASYMROTORLINES  Every line J'' <= JMax with its strength, cached by constants.
persistent Cache
Key = [A B C Au Bu Cu JMax];
if isempty(Cache)
    Cache = {};
end
for c = 1:numel(Cache)
    if isequal(Cache{c}.Key, Key)
        Lst = Cache{c}.Lst;
        if c > 1   % most recent first
            Cache = Cache([c, 1:c-1, c+1:end]);
        end
        return;
    end
end

Lo = cell(JMax + 1, 1);
Hi = cell(JMax + 2, 1);
for J = 0:JMax
    Lo{J + 1} = RotorLevels(J, A, B, C);
end
for J = 0:JMax + 1
    Hi{J + 1} = RotorLevels(J, Au, Bu, Cu);
end

Scut = 1e-12;
NuC = {}; SC = {}; TC = {}; ElC = {}; OddC = {}; LabC = {};
for Jl = 0:JMax
    L = Lo{Jl + 1};
    for Ju = max(Jl - 1, 0):Jl + 1
        if Jl == 0 && Ju == 0
            continue;
        end
        U = Hi{Ju + 1};
        dJ = Ju - Jl;
        M0 = CGMatrix(Jl, Ju, 0);
        Mp = CGMatrix(Jl, Ju, 1);
        Mm = CGMatrix(Jl, Ju, -1);
        Mt = {M0, (Mm - Mp)/sqrt(2), (Mm + Mp)/sqrt(2)};
        for bl = 1:4
            il = find(L.Blk == bl);
            if isempty(il)
                continue;
            end
            [kp, P] = BlockParity(bl);
            for Ty = 1:3
                % The upper block a transition of this type can reach:
                % CG symmetry under K -> -K, see the header.
                switch Ty
                    case 1, kpu = kp;     Pu = -P*(-1)^dJ;
                    case 2, kpu = 1 - kp; Pu =  P*(-1)^dJ;
                    case 3, kpu = 1 - kp; Pu = -P*(-1)^dJ;
                end
                bu = 2*kpu + (Pu < 0) + 1;
                iu = find(U.Blk == bu);
                if isempty(iu)
                    continue;
                end
                rl = find(mod((-Jl:Jl).', 2) == kp);
                ru = find(mod((-Ju:Ju).', 2) == kpu);
                R = full(U.V(ru, iu).' * (Mt{Ty}(ru, rl) * L.V(rl, il)));
                S = (2*Jl + 1)*R.^2;
                [f, i] = find(S > Scut);
                if isempty(f)
                    continue;
                end
                f = f(:); i = i(:);   % find on a row returns rows
                lin = sub2ind(size(S), f, i);
                fi = iu(f); ii = il(i);
                NuC{end+1}  = U.E(fi) - L.E(ii);               %#ok<AGROW>
                SC{end+1}   = reshape(S(lin), [], 1);            %#ok<AGROW>
                TC{end+1}   = Ty + zeros(numel(f), 1);           %#ok<AGROW>
                ElC{end+1}  = L.E(ii);                           %#ok<AGROW>
                OddC{end+1} = L.Odd(ii, :);                      %#ok<AGROW>
                LabC{end+1} = [Jl + zeros(numel(f), 1), L.Lab(ii, :), ...
                               Ju + zeros(numel(f), 1), U.Lab(fi, :)]; %#ok<AGROW>
            end
        end
    end
end
Lst.Nu   = vertcat(NuC{:});
Lst.S    = vertcat(SC{:});
Lst.Type = vertcat(TC{:});
Lst.Elow = vertcat(ElC{:});
Lst.Odd  = vertcat(OddC{:});
Lst.Lab  = vertcat(LabC{:});

Cache = [{struct('Key', Key, 'Lst', Lst)}, Cache];
if numel(Cache) > 4
    Cache = Cache(1:4);
end
end


% -------------------------------------------------------------------------
function L = RotorLevels(J, A, B, C)
%ROTORLEVELS  Levels of one J, sorted by energy, with exact symmetry.
%   L.E    energies (the 0_00 level at 0)
%   L.V    eigenvectors on |J,K>, K = -J..J
%   L.Blk  Wang block: 1 K even sym, 2 K even anti, 3 K odd sym, 4 K odd anti
%   L.Odd  odd under C2 about a, b, c (logical, one column each)
%   L.Lab  [Ka Kc]
K = (-J:J).';
n = 2*J + 1;
H = diag(A*K.^2 + (B + C)/2*(J*(J + 1) - K.^2));
if J >= 1
    k = K(1:end - 2);                      % <K+2|H|K>
    v = (B - C)/4*sqrt((J*(J + 1) - k.*(k + 1)).*(J*(J + 1) - (k + 1).*(k + 2)));
    H = H + diag(v, 2) + diag(v, -2);
end
% H commutes with |K> <-> |-K> (no phase), so the Wang combinations split
% it into four blocks.
E = zeros(n, 1); V = zeros(n, n); Blk = zeros(n, 1); c = 0;
for kp = 0:1
    Kpos = K(K >= 0 & mod(K, 2) == kp);
    for s = [1, -1]
        kk = Kpos(Kpos > 0 | s > 0);       % K = 0 is symmetric only
        m = numel(kk);
        if m == 0
            continue;
        end
        Ub = zeros(n, m);
        Ub(sub2ind([n, m], J + 1 + kk, (1:m).')) = 1/sqrt(2);
        Ub(sub2ind([n, m], J + 1 - kk, (1:m).')) = s/sqrt(2);
        Ub(J + 1, kk == 0) = 1;
        Hb = Ub.'*H*Ub;
        [Vb, Db] = eig((Hb + Hb.')/2);
        E(c + 1:c + m) = diag(Db);
        V(:, c + 1:c + m) = Ub*Vb;
        Blk(c + 1:c + m) = 2*kp + (s < 0) + 1;
        c = c + m;
    end
end
[E, o] = sort(E);
V = V(:, o);
Blk = Blk(o);
[kp, P] = BlockParity(Blk);
% Symmetry under the C2 rotations: about a (-1)^K; about b (-1)^J P, i.e.
% (-1)^(Ka+Kc); about c the product of the two, i.e. (-1)^Kc.
Oa = kp == 1;
Ob = mod(J, 2) ~= (P < 0);
Oc = xor(Oa, Ob);
L.Odd = [Oa, Ob, Oc];
% Ka, Kc from the energy order (tau = Ka - Kc rises with energy), then
% corrected with the exact parities where two levels are degenerate to
% rounding and their order is arbitrary (labels assume A >= B >= C).
nn = (0:n - 1).';
Ka = floor((nn + 1)/2);
Kc = J - floor(nn/2);
Bad = mod(Ka, 2) ~= Oa | mod(Kc, 2) ~= Oc;
for i = 1:n - 1
    if Bad(i) && Bad(i + 1) && mod(Ka(i + 1), 2) == Oa(i) && mod(Kc(i + 1), 2) == Oc(i) ...
            && mod(Ka(i), 2) == Oa(i + 1) && mod(Kc(i), 2) == Oc(i + 1)
        Ka([i, i + 1]) = Ka([i + 1, i]);
        Kc([i, i + 1]) = Kc([i + 1, i]);
        Bad([i, i + 1]) = false;
    end
end
L.E = E;
L.V = V;
L.Blk = Blk;
L.Lab = [Ka, Kc];
end


% -------------------------------------------------------------------------
function [kp, P] = BlockParity(Blk)
kp = double(Blk >= 3);
P = 1 - 2*double(mod(Blk - 1, 2) == 1);
end


% -------------------------------------------------------------------------
function M = CGMatrix(Jl, Ju, q)
%CGMATRIX  M(K'+Ju+1, K+Jl+1) = <Jl K; 1 q | Ju K+q>, K' = K + q.
K = (-Jl:Jl).';
Mz = K + q;                                % the M of the coupled state
ok = abs(Mz) <= Ju;
K = K(ok); Mz = Mz(ok);
j = Jl;
switch Ju - Jl
    case 1
        switch q
            case 1,  v = sqrt((j + Mz).*(j + Mz + 1)/((2*j + 1)*(2*j + 2)));
            case 0,  v = sqrt((j - Mz + 1).*(j + Mz + 1)/((2*j + 1)*(j + 1)));
            case -1, v = sqrt((j - Mz).*(j - Mz + 1)/((2*j + 1)*(2*j + 2)));
        end
    case 0
        switch q
            case 1,  v = -sqrt((j + Mz).*(j - Mz + 1)/(2*j*(j + 1)));
            case 0,  v = Mz/sqrt(j*(j + 1));
            case -1, v = sqrt((j - Mz).*(j + Mz + 1)/(2*j*(j + 1)));
        end
    case -1
        switch q
            case 1,  v = sqrt((j - Mz).*(j - Mz + 1)/(2*j*(2*j + 1)));
            case 0,  v = -sqrt((j - Mz).*(j + Mz)/(j*(2*j + 1)));
            case -1, v = sqrt((j + Mz + 1).*(j + Mz)/(2*j*(2*j + 1)));
        end
end
M = sparse(Mz + Ju + 1, K + Jl + 1, v, 2*Ju + 1, 2*Jl + 1);
end
