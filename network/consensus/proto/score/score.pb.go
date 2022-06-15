// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.28.0
// 	protoc        v3.6.1
// source: score.proto

package score

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

type Trust struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Scores map[string]float64 `protobuf:"bytes,1,rep,name=scores,proto3" json:"scores,omitempty" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"fixed64,2,opt,name=value,proto3"`
}

func (x *Trust) Reset() {
	*x = Trust{}
	if protoimpl.UnsafeEnabled {
		mi := &file_score_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Trust) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Trust) ProtoMessage() {}

func (x *Trust) ProtoReflect() protoreflect.Message {
	mi := &file_score_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Trust.ProtoReflect.Descriptor instead.
func (*Trust) Descriptor() ([]byte, []int) {
	return file_score_proto_rawDescGZIP(), []int{0}
}

func (x *Trust) GetScores() map[string]float64 {
	if x != nil {
		return x.Scores
	}
	return nil
}

type Score struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Items []float64 `protobuf:"fixed64,1,rep,packed,name=items,proto3" json:"items,omitempty"`
}

func (x *Score) Reset() {
	*x = Score{}
	if protoimpl.UnsafeEnabled {
		mi := &file_score_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Score) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Score) ProtoMessage() {}

func (x *Score) ProtoReflect() protoreflect.Message {
	mi := &file_score_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Score.ProtoReflect.Descriptor instead.
func (*Score) Descriptor() ([]byte, []int) {
	return file_score_proto_rawDescGZIP(), []int{1}
}

func (x *Score) GetItems() []float64 {
	if x != nil {
		return x.Items
	}
	return nil
}

type Similarity struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	N      uint32   `protobuf:"varint,1,opt,name=n,proto3" json:"n,omitempty"`
	Scores []*Score `protobuf:"bytes,2,rep,name=scores,proto3" json:"scores,omitempty"`
}

func (x *Similarity) Reset() {
	*x = Similarity{}
	if protoimpl.UnsafeEnabled {
		mi := &file_score_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Similarity) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Similarity) ProtoMessage() {}

func (x *Similarity) ProtoReflect() protoreflect.Message {
	mi := &file_score_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Similarity.ProtoReflect.Descriptor instead.
func (*Similarity) Descriptor() ([]byte, []int) {
	return file_score_proto_rawDescGZIP(), []int{2}
}

func (x *Similarity) GetN() uint32 {
	if x != nil {
		return x.N
	}
	return 0
}

func (x *Similarity) GetScores() []*Score {
	if x != nil {
		return x.Scores
	}
	return nil
}

var File_score_proto protoreflect.FileDescriptor

var file_score_proto_rawDesc = []byte{
	0x0a, 0x0b, 0x73, 0x63, 0x6f, 0x72, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x05, 0x73,
	0x63, 0x6f, 0x72, 0x65, 0x22, 0x74, 0x0a, 0x05, 0x54, 0x72, 0x75, 0x73, 0x74, 0x12, 0x30, 0x0a,
	0x06, 0x73, 0x63, 0x6f, 0x72, 0x65, 0x73, 0x18, 0x01, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x18, 0x2e,
	0x73, 0x63, 0x6f, 0x72, 0x65, 0x2e, 0x54, 0x72, 0x75, 0x73, 0x74, 0x2e, 0x53, 0x63, 0x6f, 0x72,
	0x65, 0x73, 0x45, 0x6e, 0x74, 0x72, 0x79, 0x52, 0x06, 0x73, 0x63, 0x6f, 0x72, 0x65, 0x73, 0x1a,
	0x39, 0x0a, 0x0b, 0x53, 0x63, 0x6f, 0x72, 0x65, 0x73, 0x45, 0x6e, 0x74, 0x72, 0x79, 0x12, 0x10,
	0x0a, 0x03, 0x6b, 0x65, 0x79, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x03, 0x6b, 0x65, 0x79,
	0x12, 0x14, 0x0a, 0x05, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x01, 0x52,
	0x05, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x3a, 0x02, 0x38, 0x01, 0x22, 0x21, 0x0a, 0x05, 0x53, 0x63,
	0x6f, 0x72, 0x65, 0x12, 0x18, 0x0a, 0x05, 0x69, 0x74, 0x65, 0x6d, 0x73, 0x18, 0x01, 0x20, 0x03,
	0x28, 0x01, 0x42, 0x02, 0x10, 0x01, 0x52, 0x05, 0x69, 0x74, 0x65, 0x6d, 0x73, 0x22, 0x40, 0x0a,
	0x0a, 0x53, 0x69, 0x6d, 0x69, 0x6c, 0x61, 0x72, 0x69, 0x74, 0x79, 0x12, 0x0c, 0x0a, 0x01, 0x6e,
	0x18, 0x01, 0x20, 0x01, 0x28, 0x0d, 0x52, 0x01, 0x6e, 0x12, 0x24, 0x0a, 0x06, 0x73, 0x63, 0x6f,
	0x72, 0x65, 0x73, 0x18, 0x02, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x0c, 0x2e, 0x73, 0x63, 0x6f, 0x72,
	0x65, 0x2e, 0x53, 0x63, 0x6f, 0x72, 0x65, 0x52, 0x06, 0x73, 0x63, 0x6f, 0x72, 0x65, 0x73, 0x42,
	0x48, 0x5a, 0x46, 0x67, 0x69, 0x74, 0x68, 0x75, 0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x4c, 0x65,
	0x6b, 0x73, 0x73, 0x61, 0x79, 0x73, 0x2f, 0x50, 0x72, 0x6f, 0x78, 0x44, 0x41, 0x47, 0x2f, 0x6e,
	0x65, 0x74, 0x77, 0x6f, 0x72, 0x6b, 0x2f, 0x73, 0x69, 0x6d, 0x75, 0x6c, 0x61, 0x74, 0x6f, 0x72,
	0x2f, 0x70, 0x65, 0x65, 0x72, 0x73, 0x2f, 0x63, 0x6c, 0x69, 0x65, 0x6e, 0x74, 0x2f, 0x70, 0x72,
	0x6f, 0x74, 0x6f, 0x2f, 0x73, 0x63, 0x6f, 0x72, 0x65, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f,
	0x33,
}

var (
	file_score_proto_rawDescOnce sync.Once
	file_score_proto_rawDescData = file_score_proto_rawDesc
)

func file_score_proto_rawDescGZIP() []byte {
	file_score_proto_rawDescOnce.Do(func() {
		file_score_proto_rawDescData = protoimpl.X.CompressGZIP(file_score_proto_rawDescData)
	})
	return file_score_proto_rawDescData
}

var file_score_proto_msgTypes = make([]protoimpl.MessageInfo, 4)
var file_score_proto_goTypes = []interface{}{
	(*Trust)(nil),      // 0: score.Trust
	(*Score)(nil),      // 1: score.Score
	(*Similarity)(nil), // 2: score.Similarity
	nil,                // 3: score.Trust.ScoresEntry
}
var file_score_proto_depIdxs = []int32{
	3, // 0: score.Trust.scores:type_name -> score.Trust.ScoresEntry
	1, // 1: score.Similarity.scores:type_name -> score.Score
	2, // [2:2] is the sub-list for method output_type
	2, // [2:2] is the sub-list for method input_type
	2, // [2:2] is the sub-list for extension type_name
	2, // [2:2] is the sub-list for extension extendee
	0, // [0:2] is the sub-list for field type_name
}

func init() { file_score_proto_init() }
func file_score_proto_init() {
	if File_score_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_score_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Trust); i {
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
		file_score_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Score); i {
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
		file_score_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Similarity); i {
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
			RawDescriptor: file_score_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   4,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_score_proto_goTypes,
		DependencyIndexes: file_score_proto_depIdxs,
		MessageInfos:      file_score_proto_msgTypes,
	}.Build()
	File_score_proto = out.File
	file_score_proto_rawDesc = nil
	file_score_proto_goTypes = nil
	file_score_proto_depIdxs = nil
}
