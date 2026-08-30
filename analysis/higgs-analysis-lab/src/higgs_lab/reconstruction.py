"""Final reconstruction implementation extracted from the supplied script.

The numerical function body is preserved verbatim for reference comparison.
Inputs must contain exactly four leptons in GeV and a calculated `mass` field.
Only events admitting two SFOS pairs are returned, in input order.
"""
import numpy as np
import awkward as ak
import vector
MZ = 91.2

def opening_dphi(phi1, phi2):
    # Wrap |delta_phi| into [0, pi]
    dphi = np.abs(phi1 - phi2)
    return np.where(dphi > np.pi, 2*np.pi - dphi, dphi)

def delta_r(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = opening_dphi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)

def calc_mass(lep_pt, lep_eta, lep_phi, lep_e):
    p4 = vector.zip({"pt": lep_pt, "eta": lep_eta, "phi": lep_phi, "E": lep_e})
    return (p4[:, 0] + p4[:, 1] + p4[:, 2] + p4[:, 3]).M

def reconstruct_z1_z2_fast(d):
    pt  = d['lep_pt']
    eta = d['lep_eta']
    phi = d['lep_phi']
    e   = d['lep_e']
    q   = d['lep_charge']
    pid = d['lep_type']

    p4 = vector.zip({'pt': pt, 'eta': eta, 'phi': phi, 'E': e})

    p4_4l = p4[:,0] + p4[:,1] + p4[:,2] + p4[:,3]
    pt4l  = ak.to_numpy(p4_4l.pt)

    vh_cart = np.stack([ak.to_numpy(p4_4l.energy), ak.to_numpy(p4_4l.px),
                        ak.to_numpy(p4_4l.py),     ak.to_numpy(p4_4l.pz)], axis=1)

    def to_cartesian(idx):
        px = ak.to_numpy(pt[:,idx]) * np.cos(ak.to_numpy(phi[:,idx]))
        py = ak.to_numpy(pt[:,idx]) * np.sin(ak.to_numpy(phi[:,idx]))
        pz = ak.to_numpy(pt[:,idx]) * np.sinh(ak.to_numpy(eta[:,idx]))
        en = ak.to_numpy(e[:,idx])
        return np.stack([en, px, py, pz], axis=1)

    def boost_to(p4vec, boost_target):
        E  = boost_target[:,0]
        bx = boost_target[:,1] / E
        by = boost_target[:,2] / E
        bz = boost_target[:,3] / E
        b2 = bx**2 + by**2 + bz**2
        gamma  = 1.0 / np.sqrt(np.maximum(1 - b2, 1e-12))
        bp     = bx*p4vec[:,1] + by*p4vec[:,2] + bz*p4vec[:,3]
        gamma2 = np.where(b2 > 0, (gamma - 1.0) / b2, 0.0)
        px_new = p4vec[:,1] + gamma2*bp*bx - gamma*bx*p4vec[:,0]
        py_new = p4vec[:,2] + gamma2*bp*by - gamma*by*p4vec[:,0]
        pz_new = p4vec[:,3] + gamma2*bp*bz - gamma*bz*p4vec[:,0]
        e_new  = gamma*(p4vec[:,0] - bp)
        return np.stack([e_new, px_new, py_new, pz_new], axis=1)

    def calc_theta(lneg, vz, vh):
        lneg_zrf = boost_to(lneg, vz)
        h_zrf    = boost_to(vh,   vz)
        z_dir = -h_zrf[:,1:4]
        z_dir = z_dir / (np.linalg.norm(z_dir, axis=1, keepdims=True) + 1e-12)
        l_dir = lneg_zrf[:,1:4]
        l_dir = l_dir / (np.linalg.norm(l_dir, axis=1, keepdims=True) + 1e-12)
        return np.arccos(np.clip(np.sum(z_dir * l_dir, axis=1), -1, 1))

    def calc_Theta(vz1, vh):
        vz1_hrf = boost_to(vz1, vh)
        z1_dir  = vz1_hrf[:,1:4]
        z1_dir  = z1_dir / (np.linalg.norm(z1_dir, axis=1, keepdims=True) + 1e-12)
        beam    = np.zeros_like(z1_dir); beam[:,2] = 1.0
        return np.arccos(np.clip(np.sum(z1_dir * beam, axis=1), -1, 1))

    def calc_phi_phi1(lneg1, lpos1, lneg2, lpos2, vz1, vh):
        lneg1_h = boost_to(lneg1, vh); lpos1_h = boost_to(lpos1, vh)
        lneg2_h = boost_to(lneg2, vh); lpos2_h = boost_to(lpos2, vh)
        vz1_h   = boost_to(vz1,   vh)

        def p3(v): return v[:,1:4]

        n1 = np.cross(p3(lneg1_h), p3(lpos1_h))
        n1 = n1 / (np.linalg.norm(n1, axis=1, keepdims=True) + 1e-12)
        n2 = np.cross(p3(lneg2_h), p3(lpos2_h))
        n2 = n2 / (np.linalg.norm(n2, axis=1, keepdims=True) + 1e-12)
        z1_dir = p3(vz1_h)
        z1_dir = z1_dir / (np.linalg.norm(z1_dir, axis=1, keepdims=True) + 1e-12)

        cos_phi  = np.clip(np.sum(n1 * n2, axis=1), -1, 1)
        sign_phi = np.sign(np.sum(n1 * np.cross(n2, z1_dir), axis=1))
        Phi      = sign_phi * np.arccos(cos_phi)

        beam    = np.zeros_like(z1_dir); beam[:,2] = 1.0
        n_prod  = np.cross(z1_dir, beam)
        norm_np = np.linalg.norm(n_prod, axis=1, keepdims=True)
        n_prod  = np.where(norm_np > 1e-6, n_prod / (norm_np + 1e-12), n_prod)
        cos_phi1  = np.clip(np.sum(n1 * n_prod, axis=1), -1, 1)
        sign_phi1 = np.sign(np.sum(n_prod * np.cross(n1, z1_dir), axis=1))
        Phi1      = sign_phi1 * np.arccos(cos_phi1)

        return Phi, Phi1

    pairings = [(0,1,2,3), (0,2,1,3), (0,3,1,2)]
    results  = []

    for (i,j,k,l) in pairings:
        sfos_1 = (pid[:,i] == pid[:,j]) & (q[:,i] + q[:,j] == 0)
        sfos_2 = (pid[:,k] == pid[:,l]) & (q[:,k] + q[:,l] == 0)
        valid  = sfos_1 & sfos_2

        p4_z1 = p4[:,i] + p4[:,j]
        p4_z2 = p4[:,k] + p4[:,l]
        m_ij  = p4_z1.M
        m_kl  = p4_z2.M

        ij_is_Z1 = np.abs(ak.to_numpy(m_ij) - MZ) <= np.abs(ak.to_numpy(m_kl) - MZ)

        pt_i,  pt_j  = ak.to_numpy(pt[:,i]),  ak.to_numpy(pt[:,j])
        pt_k,  pt_l  = ak.to_numpy(pt[:,k]),  ak.to_numpy(pt[:,l])
        eta_i, eta_j = ak.to_numpy(eta[:,i]), ak.to_numpy(eta[:,j])
        eta_k, eta_l = ak.to_numpy(eta[:,k]), ak.to_numpy(eta[:,l])
        phi_i, phi_j = ak.to_numpy(phi[:,i]), ak.to_numpy(phi[:,j])
        phi_k, phi_l = ak.to_numpy(phi[:,k]), ak.to_numpy(phi[:,l])

        i_is_lead_ij = pt_i >= pt_j
        k_is_lead_kl = pt_k >= pt_l

        vz1_cart = np.stack([ak.to_numpy(p4_z1.energy), ak.to_numpy(p4_z1.px),
                              ak.to_numpy(p4_z1.py),    ak.to_numpy(p4_z1.pz)], axis=1)
        vz2_cart = np.stack([ak.to_numpy(p4_z2.energy), ak.to_numpy(p4_z2.px),
                              ak.to_numpy(p4_z2.py),    ak.to_numpy(p4_z2.pz)], axis=1)

        vi = to_cartesian(i); vj = to_cartesian(j)
        vk = to_cartesian(k); vl = to_cartesian(l)

        qi = ak.to_numpy(q[:,i])
        qk = ak.to_numpy(q[:,k])
        lneg_ij = np.where(qi[:,None] < 0, vi, vj)
        lpos_ij = np.where(qi[:,None] >= 0, vi, vj)
        lneg_kl = np.where(qk[:,None] < 0, vk, vl)
        lpos_kl = np.where(qk[:,None] >= 0, vk, vl)

        theta1_ij = calc_theta(lneg_ij, vz1_cart, vh_cart)
        theta1_kl = calc_theta(lneg_kl, vz2_cart, vh_cart)
        theta1    = np.where(ij_is_Z1, theta1_ij, theta1_kl)
        theta2    = np.where(ij_is_Z1, theta1_kl, theta1_ij)

        vz1_final = np.where(ij_is_Z1[:,None], vz1_cart, vz2_cart)
        vz2_final = np.where(ij_is_Z1[:,None], vz2_cart, vz1_cart)

        Theta = calc_Theta(vz1_final, vh_cart)

        lneg_z1 = np.where(ij_is_Z1[:,None], lneg_ij, lneg_kl)
        lpos_z1 = np.where(ij_is_Z1[:,None], lpos_ij, lpos_kl)
        lneg_z2 = np.where(ij_is_Z1[:,None], lneg_kl, lneg_ij)
        lpos_z2 = np.where(ij_is_Z1[:,None], lpos_kl, lpos_ij)
        
        Phi, Phi1 = calc_phi_phi1(lneg_z1, lpos_z1, lneg_z2, lpos_z2,
                                   vz1_final, vh_cart)

        dphi_z1 = opening_dphi(phi_i, phi_j)
        dphi_z2 = opening_dphi(phi_k, phi_l)

        # All 6 inter-lepton dR pairs (raw, before Z1/Z2 assignment)
        dR_ij = delta_r(eta_i, phi_i, eta_j, phi_j)
        dR_ik = delta_r(eta_i, phi_i, eta_k, phi_k)
        dR_il = delta_r(eta_i, phi_i, eta_l, phi_l)
        dR_jk = delta_r(eta_j, phi_j, eta_k, phi_k)
        dR_jl = delta_r(eta_j, phi_j, eta_l, phi_l)
        dR_kl = delta_r(eta_k, phi_k, eta_l, phi_l)

        # ── Correct cross-pair dR after Z1/Z2 assignment ──────────────────
        # Lepton ordering within each Z: lead (0), sublead (1)
        # So the 4 assigned leptons are: Z1_lead(0), Z1_sub(1), Z2_lead(2), Z2_sub(3)
        # and the cross pairs are: 02, 03, 12, 13

        # eta/phi of each assigned lepton
        eta_z1_lead = np.where(ij_is_Z1,
                               np.where(i_is_lead_ij, eta_i, eta_j),
                               np.where(k_is_lead_kl, eta_k, eta_l))
        eta_z1_sub  = np.where(ij_is_Z1,
                               np.where(i_is_lead_ij, eta_j, eta_i),
                               np.where(k_is_lead_kl, eta_l, eta_k))
        eta_z2_lead = np.where(ij_is_Z1,
                               np.where(k_is_lead_kl, eta_k, eta_l),
                               np.where(i_is_lead_ij, eta_i, eta_j))
        eta_z2_sub  = np.where(ij_is_Z1,
                               np.where(k_is_lead_kl, eta_l, eta_k),
                               np.where(i_is_lead_ij, eta_j, eta_i))

        phi_z1_lead = np.where(ij_is_Z1,
                               np.where(i_is_lead_ij, phi_i, phi_j),
                               np.where(k_is_lead_kl, phi_k, phi_l))
        phi_z1_sub  = np.where(ij_is_Z1,
                               np.where(i_is_lead_ij, phi_j, phi_i),
                               np.where(k_is_lead_kl, phi_l, phi_k))
        phi_z2_lead = np.where(ij_is_Z1,
                               np.where(k_is_lead_kl, phi_k, phi_l),
                               np.where(i_is_lead_ij, phi_i, phi_j))
        phi_z2_sub  = np.where(ij_is_Z1,
                               np.where(k_is_lead_kl, phi_l, phi_k),
                               np.where(i_is_lead_ij, phi_j, phi_i))

        dR_02 = delta_r(eta_z1_lead, phi_z1_lead, eta_z2_lead, phi_z2_lead)
        dR_03 = delta_r(eta_z1_lead, phi_z1_lead, eta_z2_sub,  phi_z2_sub)
        dR_12 = delta_r(eta_z1_sub,  phi_z1_sub,  eta_z2_lead, phi_z2_lead)
        dR_13 = delta_r(eta_z1_sub,  phi_z1_sub,  eta_z2_sub,  phi_z2_sub)

        results.append({
            'valid':            valid,
            'mz1':              np.where(ij_is_Z1, ak.to_numpy(m_ij),     ak.to_numpy(m_kl)),
            'mz2':              np.where(ij_is_Z1, ak.to_numpy(m_kl),     ak.to_numpy(m_ij)),
            'ptz1':             np.where(ij_is_Z1, ak.to_numpy(p4_z1.pt), ak.to_numpy(p4_z2.pt)),
            'ptz2':             np.where(ij_is_Z1, ak.to_numpy(p4_z2.pt), ak.to_numpy(p4_z1.pt)),
            'pt_z1_l1':         np.where(ij_is_Z1, np.maximum(pt_i, pt_j), np.maximum(pt_k, pt_l)),
            'pt_z1_l2':         np.where(ij_is_Z1, np.minimum(pt_i, pt_j), np.minimum(pt_k, pt_l)),
            'pt_z2_l1':         np.where(ij_is_Z1, np.maximum(pt_k, pt_l), np.maximum(pt_i, pt_j)),
            'pt_z2_l2':         np.where(ij_is_Z1, np.minimum(pt_k, pt_l), np.minimum(pt_i, pt_j)),
            'pt4l':             pt4l,
            'eta_z1_l1':        eta_z1_lead,
            'eta_z1_l2':        eta_z1_sub,
            'eta_z2_l1':        eta_z2_lead,
            'eta_z2_l2':        eta_z2_sub,
            'theta1':           theta1,
            'theta2':           theta2,
            'Theta':            Theta,
            'Phi':              Phi,
            'Phi1':             Phi1,
            'dphi_z1':          np.where(ij_is_Z1, dphi_z1, dphi_z2),
            'dphi_z2':          np.where(ij_is_Z1, dphi_z2, dphi_z1),
            'dR_z1':            np.where(ij_is_Z1, dR_ij,   dR_kl),
            'dR_z2':            np.where(ij_is_Z1, dR_kl,   dR_ij),
            'dR_02':            dR_02,
            'dR_03':            dR_03,
            'dR_12':            dR_12,
            'dR_13':            dR_13,
            'scalar_pt_sum_z1': np.where(ij_is_Z1, pt_i + pt_j, pt_k + pt_l),
            'scalar_pt_sum_z2': np.where(ij_is_Z1, pt_k + pt_l, pt_i + pt_j),
            'type_z1_l1': np.where(ij_is_Z1,
                              np.where(i_is_lead_ij, ak.to_numpy(pid[:,i]), ak.to_numpy(pid[:,j])),
                              np.where(k_is_lead_kl, ak.to_numpy(pid[:,k]), ak.to_numpy(pid[:,l]))),
            'type_z1_l2': np.where(ij_is_Z1,
                              np.where(i_is_lead_ij, ak.to_numpy(pid[:,j]), ak.to_numpy(pid[:,i])),
                              np.where(k_is_lead_kl, ak.to_numpy(pid[:,l]), ak.to_numpy(pid[:,k]))),
            'type_z2_l1': np.where(ij_is_Z1,
                              np.where(k_is_lead_kl, ak.to_numpy(pid[:,k]), ak.to_numpy(pid[:,l])),
                              np.where(i_is_lead_ij, ak.to_numpy(pid[:,i]), ak.to_numpy(pid[:,j]))),
            'type_z2_l2': np.where(ij_is_Z1,
                              np.where(k_is_lead_kl, ak.to_numpy(pid[:,l]), ak.to_numpy(pid[:,k])),
                              np.where(i_is_lead_ij, ak.to_numpy(pid[:,j]), ak.to_numpy(pid[:,i]))),
            'jet_n':   ak.to_numpy(d['jet_n']),
            'met':     ak.to_numpy(d['met']),
            'met_phi': ak.to_numpy(d['met_phi']),
            'mass':    ak.to_numpy(d['mass']),
        })

    keys = ['mz1','mz2','ptz1','ptz2','pt4l',
            'pt_z1_l1','pt_z1_l2','pt_z2_l1','pt_z2_l2',
            'eta_z1_l1','eta_z1_l2','eta_z2_l1','eta_z2_l2',
            'theta1','theta2','Theta','Phi','Phi1',
            'dphi_z1','dphi_z2',
            'dR_z1','dR_z2','dR_02','dR_03','dR_12','dR_13',
            'scalar_pt_sum_z1','scalar_pt_sum_z2',
            'type_z1_l1','type_z1_l2','type_z2_l1','type_z2_l2',
            'jet_n','met','met_phi','mass']

    stacked   = {k: np.stack([r[k] for r in results], axis=1) for k in keys}
    val_arr   = np.stack([ak.to_numpy(r['valid']) for r in results], axis=1)

    dist      = np.where(val_arr, np.abs(stacked['mz1'] - MZ), np.inf)
    best_idx  = np.argmin(dist, axis=1)
    row_idx   = np.arange(len(best_idx))
    valid_any = val_arr.any(axis=1)

    out = {k: stacked[k][row_idx, best_idx][valid_any] for k in keys}
    out['w'] = ak.to_numpy(d['totalWeight'])[valid_any] if 'totalWeight' in d.fields else np.ones(valid_any.sum())
    return out
