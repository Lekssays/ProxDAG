# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modelUpdate.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='modelUpdate.proto',
  package='modelUpdate',
  syntax='proto3',
  serialized_options=_b('ZLgithub.com/Lekssays/ProxDAG/network/simulator/peers/client/proto/modelUpdate'),
  serialized_pb=_b('\n\x11modelUpdate.proto\x12\x0bmodelUpdate\"R\n\x0bModelUpdate\x12\x0f\n\x07modelID\x18\x01 \x01(\t\x12\x0f\n\x07parents\x18\x02 \x03(\t\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\x0c\x12\x10\n\x08\x65ndpoint\x18\x04 \x01(\tBNZLgithub.com/Lekssays/ProxDAG/network/simulator/peers/client/proto/modelUpdateb\x06proto3')
)




_MODELUPDATE = _descriptor.Descriptor(
  name='ModelUpdate',
  full_name='modelUpdate.ModelUpdate',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='modelID', full_name='modelUpdate.ModelUpdate.modelID', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='parents', full_name='modelUpdate.ModelUpdate.parents', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='content', full_name='modelUpdate.ModelUpdate.content', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='endpoint', full_name='modelUpdate.ModelUpdate.endpoint', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=34,
  serialized_end=116,
)

DESCRIPTOR.message_types_by_name['ModelUpdate'] = _MODELUPDATE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ModelUpdate = _reflection.GeneratedProtocolMessageType('ModelUpdate', (_message.Message,), dict(
  DESCRIPTOR = _MODELUPDATE,
  __module__ = 'modelUpdate_pb2'
  # @@protoc_insertion_point(class_scope:modelUpdate.ModelUpdate)
  ))
_sym_db.RegisterMessage(ModelUpdate)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)