// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.28.0
// 	protoc        v3.6.1
// source: modelUpdate.proto

package modelUpdate

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type ModelUpdate struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	ModelID   string   `protobuf:"bytes,1,opt,name=modelID,proto3" json:"modelID,omitempty"`
	Parents   []string `protobuf:"bytes,2,rep,name=parents,proto3" json:"parents,omitempty"`
	Weights   string   `protobuf:"bytes,3,opt,name=weights,proto3" json:"weights,omitempty"`
	Pubkey    string   `protobuf:"bytes,4,opt,name=pubkey,proto3" json:"pubkey,omitempty"`
	Timestamp uint32   `protobuf:"varint,5,opt,name=timestamp,proto3" json:"timestamp,omitempty"`
	Accuracy  float32  `protobuf:"fixed32,6,opt,name=accuracy,proto3" json:"accuracy,omitempty"`
}

func (x *ModelUpdate) Reset() {
	*x = ModelUpdate{}
	if protoimpl.UnsafeEnabled {
		mi := &file_modelUpdate_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ModelUpdate) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelUpdate) ProtoMessage() {}

func (x *ModelUpdate) ProtoReflect() protoreflect.Message {
	mi := &file_modelUpdate_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelUpdate.ProtoReflect.Descriptor instead.
func (*ModelUpdate) Descriptor() ([]byte, []int) {
	return file_modelUpdate_proto_rawDescGZIP(), []int{0}
}

func (x *ModelUpdate) GetModelID() string {
	if x != nil {
		return x.ModelID
	}
	return ""
}

func (x *ModelUpdate) GetParents() []string {
	if x != nil {
		return x.Parents
	}
	return nil
}

func (x *ModelUpdate) GetWeights() string {
	if x != nil {
		return x.Weights
	}
	return ""
}

func (x *ModelUpdate) GetPubkey() string {
	if x != nil {
		return x.Pubkey
	}
	return ""
}

func (x *ModelUpdate) GetTimestamp() uint32 {
	if x != nil {
		return x.Timestamp
	}
	return 0
}

func (x *ModelUpdate) GetAccuracy() float32 {
	if x != nil {
		return x.Accuracy
	}
	return 0
}

var File_modelUpdate_proto protoreflect.FileDescriptor

var file_modelUpdate_proto_rawDesc = []byte{
	0x0a, 0x11, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x55, 0x70, 0x64, 0x61, 0x74, 0x65, 0x2e, 0x70, 0x72,
	0x6f, 0x74, 0x6f, 0x12, 0x0b, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x55, 0x70, 0x64, 0x61, 0x74, 0x65,
	0x22, 0xad, 0x01, 0x0a, 0x0b, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x55, 0x70, 0x64, 0x61, 0x74, 0x65,
	0x12, 0x18, 0x0a, 0x07, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x49, 0x44, 0x18, 0x01, 0x20, 0x01, 0x28,
	0x09, 0x52, 0x07, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x49, 0x44, 0x12, 0x18, 0x0a, 0x07, 0x70, 0x61,
	0x72, 0x65, 0x6e, 0x74, 0x73, 0x18, 0x02, 0x20, 0x03, 0x28, 0x09, 0x52, 0x07, 0x70, 0x61, 0x72,
	0x65, 0x6e, 0x74, 0x73, 0x12, 0x18, 0x0a, 0x07, 0x77, 0x65, 0x69, 0x67, 0x68, 0x74, 0x73, 0x18,
	0x03, 0x20, 0x01, 0x28, 0x09, 0x52, 0x07, 0x77, 0x65, 0x69, 0x67, 0x68, 0x74, 0x73, 0x12, 0x16,
	0x0a, 0x06, 0x70, 0x75, 0x62, 0x6b, 0x65, 0x79, 0x18, 0x04, 0x20, 0x01, 0x28, 0x09, 0x52, 0x06,
	0x70, 0x75, 0x62, 0x6b, 0x65, 0x79, 0x12, 0x1c, 0x0a, 0x09, 0x74, 0x69, 0x6d, 0x65, 0x73, 0x74,
	0x61, 0x6d, 0x70, 0x18, 0x05, 0x20, 0x01, 0x28, 0x0d, 0x52, 0x09, 0x74, 0x69, 0x6d, 0x65, 0x73,
	0x74, 0x61, 0x6d, 0x70, 0x12, 0x1a, 0x0a, 0x08, 0x61, 0x63, 0x63, 0x75, 0x72, 0x61, 0x63, 0x79,
	0x18, 0x06, 0x20, 0x01, 0x28, 0x02, 0x52, 0x08, 0x61, 0x63, 0x63, 0x75, 0x72, 0x61, 0x63, 0x79,
	0x42, 0x3e, 0x5a, 0x3c, 0x67, 0x69, 0x74, 0x68, 0x75, 0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x4c,
	0x65, 0x6b, 0x73, 0x73, 0x61, 0x79, 0x73, 0x2f, 0x50, 0x72, 0x6f, 0x78, 0x44, 0x41, 0x47, 0x2f,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x63, 0x6f, 0x6c, 0x2f, 0x67, 0x72, 0x61, 0x70, 0x68, 0x2f, 0x70,
	0x72, 0x6f, 0x74, 0x6f, 0x2f, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x55, 0x70, 0x64, 0x61, 0x74, 0x65,
	0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_modelUpdate_proto_rawDescOnce sync.Once
	file_modelUpdate_proto_rawDescData = file_modelUpdate_proto_rawDesc
)

func file_modelUpdate_proto_rawDescGZIP() []byte {
	file_modelUpdate_proto_rawDescOnce.Do(func() {
		file_modelUpdate_proto_rawDescData = protoimpl.X.CompressGZIP(file_modelUpdate_proto_rawDescData)
	})
	return file_modelUpdate_proto_rawDescData
}

var file_modelUpdate_proto_msgTypes = make([]protoimpl.MessageInfo, 1)
var file_modelUpdate_proto_goTypes = []interface{}{
	(*ModelUpdate)(nil), // 0: modelUpdate.ModelUpdate
}
var file_modelUpdate_proto_depIdxs = []int32{
	0, // [0:0] is the sub-list for method output_type
	0, // [0:0] is the sub-list for method input_type
	0, // [0:0] is the sub-list for extension type_name
	0, // [0:0] is the sub-list for extension extendee
	0, // [0:0] is the sub-list for field type_name
}

func init() { file_modelUpdate_proto_init() }
func file_modelUpdate_proto_init() {
	if File_modelUpdate_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_modelUpdate_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ModelUpdate); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_modelUpdate_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   1,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_modelUpdate_proto_goTypes,
		DependencyIndexes: file_modelUpdate_proto_depIdxs,
		MessageInfos:      file_modelUpdate_proto_msgTypes,
	}.Build()
	File_modelUpdate_proto = out.File
	file_modelUpdate_proto_rawDesc = nil
	file_modelUpdate_proto_goTypes = nil
	file_modelUpdate_proto_depIdxs = nil
}