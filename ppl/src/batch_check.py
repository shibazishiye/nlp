import onnxruntime as ort

sess = ort.InferenceSession("roberta_wwm_mlm_dynamic_new.onnx", providers=["CPUExecutionProvider"])

print("Inputs:")
for i in sess.get_inputs():
    print(i.name, i.shape)

print("\nOutputs:")
for o in sess.get_outputs():
    print(o.name, o.shape)
