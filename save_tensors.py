# from tokenizers import ByteLevelBPETokenizer, trainers, processors
# tokenizer = ByteLevelBPETokenizer()
# tokenizer.train("")
import torch


from process import get_emotion_data, get_motor_data, get_language_data, get_relational_data, get_social_data, get_gambling_data, get_wm_data

print("saving emotion tensors")
elr, erl, e_labels = get_emotion_data()
torch.save(elr, "./tensors/elr.pt") # [906, 368, 176]
torch.save(erl, "./tensors/erl.pt")
torch.save(e_labels, "./tensors/emotion_labels.pt")

print("saving motor tensors")
mlr, mrl, m_labels = get_motor_data()
torch.save(mlr, "./tensors/mlr.pt")
torch.save(mrl, "./tensors/mrl.pt")
torch.save(m_labels, "./tensors/motor_labels.pt")

print("saving language tensors")
llr, lrl, l_labels = get_language_data()
torch.save(llr, "./tensors/llr.pt")
torch.save(lrl, "./tensors/lrl.pt")
torch.save(l_labels, "./tensors/language_labels.pt")

print("saving relational tensors")
rlr, rrl, r_labels = get_relational_data()
torch.save(rlr, "./tensors/rlr.pt")
torch.save(rrl, "./tensors/rrl.pt")
torch.save(r_labels, "./tensors/relational_labels.pt")

print("saving social tensors")
slr, srl, s_labels = get_social_data()
torch.save(slr, "./tensors/slr.pt")
torch.save(srl, "./tensors/srl.pt")
torch.save(s_labels, "./tensors/social_labels.pt")

print("saving gambling tensors")
glr, grl, g_labels = get_gambling_data()
torch.save(glr, "./tensors/glr.pt")
torch.save(grl, "./tensors/grl.pt")
torch.save(g_labels, "./tensors/gambling_labels.pt")

print("saving wm tensors")
wlr, wrl, w_labels = get_wm_data()
torch.save(wlr, "./tensors/wlr.pt")
torch.save(wrl, "./tensors/wrl.pt")
torch.save(w_labels, "./tensors/wm_labels.pt")