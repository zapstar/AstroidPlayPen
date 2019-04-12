"""
PyLint protobuf transform plugin
"""
import enum
import logging
import os

from typing import Any, Dict, Generator, List, Optional, Union

import astroid

# Type variable to describe scalar values in protobuf
Scalar = Union[float, int, bool, bytes, str]
# Type variable to denote module globals
ModuleGlobals = Dict[str, List[astroid.Assign]]

# Default log level
DEFAULT_LOG_LEVEL = logging.ERROR

# Tab character is 4 spaces, used by our code generator
TAB = 4 * ' '

# The list of modules to be used by our code generator
IMPORTS = ['enum']


class PBFieldType(enum.Enum):
    """
    Protobuf field enum type
    """
    SCALAR = 1
    COMPLEX = 2


class PBField(object):
    """
    Class to represent fields inside a Protobuf class
    """
    def __init__(self, name, repeated, field_type, value):
        # type: (str, bool, PBFieldType, FieldValue) -> None
        """
        Constructor for the `Field` class

        :param name: Name of the field
        :param repeated: Field is repeated?
        :param field_type: Scalar/Complex?
        :param value: Value to be put in the field
        """
        self.name = name
        self.repeated = repeated
        self.type = field_type
        self.value = value

    def __repr__(self):
        # type: () -> str
        """
        Constructor representation of the current `PBField` object

        :return: Constructor representation of the current object
        """

        return '{class_name}(name={name}, repeated={repeated}, ' \
               'field_type={field_type}, value={value})' \
               ''.format(class_name=type(self).__name__,
                         name=repr(self.name),
                         repeated=repr(self.repeated),
                         field_type=str(self.type),
                         value=repr(self.value))


class PBEnumField(PBField):
    """
    Class to represent Protobuf enumerations
    """
    def __init__(self, name, repeated, value):
        # type: (str, bool, FieldValue) -> None
        """
        Constructor for the `PBEnumField` class

        :param name: Name of the field
        :param repeated: Field is repeated?
        :param value: Value to be put in the field
        """
        super(PBEnumField, self).__init__(name,
                                          repeated,
                                          PBFieldType.SCALAR,
                                          value)

    def __str__(self):
        """
        The string representation - Used to build fake
        class that will parsed by `astroid`.

        :return: String to be placed inside the `enum.Enum` class
        """
        return '{} = {}'.format(self.name, self.value)


class PBMessageField(PBField):
    """
    Class to represent Protobuf messages
    """
    def __str__(self):
        """
        The string representation of a message field. This is used
        to build a fake class that is parsed by `astroid`.
        :return: String to be placed in a class
        """
        if self.type == PBFieldType.SCALAR:
            # The field's value is a scalar value
            return 'self.{name} = {lb}{value}{rb}'\
                   ''.format(name=self.name,
                             lb='[' if self.repeated else '',
                             value=repr(self.value),
                             rb=']' if self.repeated else '')

        if self.type == PBFieldType.COMPLEX:
            # The field's value is a complex value
            if isinstance(self.value, PBMessageClass):
                # It's field value is a Message class
                return 'self.{name} = {lb}{value}(){rb}'\
                       ''.format(name=self.name,
                                 lb='[' if self.repeated else '',
                                 value=self.value.name,
                                 rb=']' if self.repeated else '')
            if isinstance(self.value, PBEnumClass):
                # It's field value is a Enum class
                enum_cls = self.value
                if enum_cls.fields:
                    # Generate the value by concatenating the class
                    # name and the first field's name
                    field_value = '{}.{}'.format(enum_cls.name,
                                                 enum_cls.fields[0].name)
                else:
                    # Protobuf promises that an enum value is
                    # compatible with an integer
                    field_value = 0
                return 'self.{name} = {lb}{value}{rb}'\
                       ''.format(name=self.name,
                                 lb='[' if self.repeated else '',
                                 value=field_value,
                                 rb=']' if self.repeated else '')

            logging.error('Message field %s has a complex value '
                          'that is neither Enum or Message', self.name)
            return ''
        logging.error('Message field %s is neither '
                      'scalar nor complex', self.name)
        return ''


class PBClassType(enum.Enum):
    """
    Protobuf class enum type
    """
    MESSAGE = 1
    ENUM = 2


class PBClass(object):
    """
    Class to represent a Protobuf class
    """
    def __init__(self, name, class_type, desc_name, fields=None):
        # type: (str, PBClassType, str, Optional[List[PBField]]) -> None
        """
        Constructor for the `PBClass` class

        :param name: Name of the class
        :param class_type: Enum/Message class type?
        :param desc_name: Descriptor name for this class
        :param fields: List of fields present in the class
        """
        if fields is None:
            # type: List[PBField]
            fields = []
        self.name = name
        self.type = class_type
        self.desc_name = desc_name
        self.fields = fields

    def str_fields(self, tab_count=1):
        # type: (int) -> str
        """
        Convert the list of fields into string and pad it
        with appropriate number of tabs in the left.

        :param tab_count: Number of tabs to insert in the left
        :return: The concatenated string we generate
        """
        field_str_list = []
        for field in self.fields:
            field_str_list.append((tab_count * TAB) + str(field))
        # Return the concatenated string
        return '\n'.join(field_str_list) if field_str_list else 'pass'

    def __repr__(self):
        # type: () -> str
        """
        Constructor representation of the current `PBClass` object

        :return: Constructor representation of the current object
        """
        return "{class_name}(name={name}, class_type={class_type}, " \
               "desc_name={desc_name}, fields=[{fields}])" \
               "".format(class_name=type(self).__name__,
                         name=repr(self.name),
                         class_type=str(self.type),
                         desc_name=repr(self.desc_name),
                         fields=', '.join(repr(f) for f in self.fields))


class PBEnumClass(PBClass):
    """
    Class to represent Protobuf enumeration class
    """
    def __init__(self, name, desc_name, fields=None):
        # type: (str, str, Optional[List[PBField]]) -> None
        """
        Constructor for the `PBEnumClass` class

        :param name: Name of the class
        :param desc_name: Descriptor name for this class
        :param fields: List of fields present in the class
        """
        super(PBEnumClass, self).__init__(name, PBClassType.ENUM,
                                          desc_name, fields)

    def __str__(self):
        return """class {}(enum.Enum):
{}""".format(self.name, self.str_fields())


class PBMessageClass(PBClass):
    """
    Class to represent Protobuf message class
    """
    def __init__(self, name, desc_name, fields=None):
        # type: (str, str, Optional[List[PBField]]) -> None
        """
        Constructor for the `PBMessageClass` class

        :param name: Name of the class
        :param desc_name: Descriptor name for this class
        :param fields: List of fields present in the class
        """
        super(PBMessageClass, self).__init__(name, PBClassType.MESSAGE,
                                             desc_name, fields)

    def __str__(self):
        return """class {}(object):
    def __init__(self):
{}""".format(self.name, self.str_fields(tab_count=2))


# Type variable to denote all field values to protobuf
FieldValue = Union[Scalar, PBClass]


def _likely_classes(module_globals):
    # type: (ModuleGlobals) -> Generator[str, None, None]
    """
    Extract likely Protobuf class names from all global variables
    * Ignore any global variable starting with `_`.
    * Should not be `sys` or `DESCRIPTOR`.

    :param module_globals: Astroid
    :return:
    """
    # A list of ignored global variable names
    ignored_vars = {'sys', 'DESCRIPTOR'}
    # Run through the global variables, eliminate all those
    # that don't meet our criteria.
    for k in module_globals:
        if k.startswith('_') or k in ignored_vars:
            continue
        logging.debug('Yielding likely class %s', k)
        yield k


def _extract_enum_descriptor_name(call):
    # type: (astroid.Call) -> Optional[str]
    """
    Given an astroid call object of `EnumTypeWrapper`
    extract the name of the `enum` descriptor.

    For example: If the `call` belongs to

    `Color = enum_type_wrapper.EnumTypeWrapper(_COLOR)`

    This function will return `_COLOR`

    :param call: An astroid.Call object belonging to `EnumTypeWrapper`
    :return: Descriptor name on success, `None` on failure
    """
    # Sanity check on the number of arguments present in `astroid.Call`
    if not call.args:
        logging.warning('No arguments present in astroid.Call %s', call)
        return None
    # Get the first argument present in `astroid.Call`, it should ideally
    # be a descriptor object
    descriptor = call.args[0]
    # Sanity check on the type of argument present in `astroid.Call`
    if not isinstance(descriptor, astroid.Name):
        logging.warning('Argument for call is not '
                        'astroid.Name: %s', descriptor)
        return None
    # Done, we've found it return the descriptor name.
    logging.debug('Call %s has the descriptor name: %s', call, descriptor.name)
    return descriptor.name


def _extract_message_descriptor_name(call):
    # type: (astroid.Call) -> Optional[str]
    """
    Given an `astroid.Call` object of `GeneratedProtocolMessageType`
    extract the name of the `message` descriptor.

    For example if `call` belongs to:

    Outer = _reflection.GeneratedProtocolMessageType('Outer', \
        (_message.Message,), dict(DESCRIPTOR = _OUTER, __module__ = 'foo_pb2'))

    This function will return `_OUTER`

    :param call: An `astroid.Call` object to `GeneratedProtocolMessageType`
    :return: Descriptor name on success, None on failure
    """
    # Sanity check on the number of arguments to `astroid.Call` object
    if len(call.args) < 3:
        logging.warning('Len of arguments for %s < 3: %s', call, call.args)
        return None
    # Get the argument for the `dict()` call in `astroid.Call` object
    dict_arg = call.args[2]
    # Extract the descriptor name object
    descriptor = None
    if isinstance(dict_arg, astroid.Call):
        # Argument defined the dictionary as dict(a=b, c=d)
        for keyword in dict_arg.keywords:
            if keyword.arg == 'DESCRIPTOR':
                descriptor = keyword.value
    elif isinstance(dict_arg, astroid.Dict):
        # Argument defined the dictionary as {a: b, c: d}
        for key, var in dict_arg.items():
            if key.value == 'DESCRIPTOR':
                descriptor = var
    else:
        # Unknown 3rd argument passed to `astroid.Call`
        logging.warning('Unknown type for dict_arg: %s', type(dict_arg))
        return None
    # Sanity check on whether or not we were able to get the descriptor?
    if descriptor is None:
        logging.warning('Descriptor not found')
        return None
    # Sanity check on the type of the descriptor object.
    if not isinstance(descriptor, astroid.Name):
        logging.warning('Descriptor is not an astroid Name')
        return None
    # Done, return the descriptor name
    logging.debug('Call %s has descriptor name: %s', call, descriptor.name)
    return descriptor.name


def _extract_classes(module_globals):
    # type: (ModuleGlobals) -> List[PBClass]
    """
    Extract list of Protobuf classes (messages, enums, ...)

    :param module_globals: List of module level globals (astroid) objects
    :return: list of protobuf classes
    """
    classes = []
    for cls_name in _likely_classes(module_globals):
        try:
            # type: List[astroid.AssignName]
            assignment_line = module_globals[cls_name]
        except KeyError:
            logging.warning('Class %s not present in module globals', cls_name)
            continue

        assign_node = assignment_line[0]
        if not isinstance(assign_node, astroid.AssignName):
            logging.debug('Invalid type for `node`: %s', type(assign_node))
            continue
        call_node = assign_node.parent.value
        if not isinstance(call_node, astroid.Call):
            logging.debug('Unknown type for `call`: %s', type(call_node))
            continue
        if call_node.func.attrname == 'EnumTypeWrapper':
            # type: Optional[str]
            desc_name = _extract_enum_descriptor_name(call_node)
            if desc_name:
                classes.append(PBEnumClass(cls_name, desc_name))
        elif call_node.func.attrname == 'GeneratedProtocolMessageType':
            # type: Optional[str]
            desc_name = _extract_message_descriptor_name(call_node)
            if desc_name:
                classes.append(PBMessageClass(cls_name, desc_name))
        else:
            logging.debug('Unknown function call: %s', call_node.func.attrname)
    return classes


FIELD_TYPES = {
    1: float(0),  # double
    2: float(0),  # float
    3: int(0),  # int64
    4: int(0),  # uint64
    5: int(0),  # int32
    6: int(0),  # fixed64
    7: int(0),  # fixed32
    8: False,  # bool
    9: '',  # string
    11: None,  # Sub-message
    12: bytes(),  # bytes
    13: int(0),  # uint32
    14: None,  # enum
    15: int(0),  # sfixed32
    16: int(0),  # sfixed64
    17: int(0),  # sint32
    18: int(0),  # sint64
}

COMPLEX_FIELDS = {
    11,  # Protobuf Message
    14,  # Protobuf Enum
}


def _extract_enum_field(call):
    # type: (astroid.Call) -> Optional[PBField]
    """
    Extract enum fields from the `EnumValueDescriptor` astroid object

    :param call: Reference to the `astroid.Call` object
    :return: A `PBField` object on success, None on failure
    """
    field_name = None
    field_number = None
    for keyword in call.keywords:
        if keyword.arg == 'name':
            field_name = getattr(keyword.value, 'value', None)
        elif keyword.arg == 'number':
            field_number = next(keyword.value.infer()).value
    # Sanity check on whether we were able to extract the name and value?
    if field_name is None:
        logging.warning('Unable to extract enum field name: %s', call)
        return None
    if field_number is None:
        logging.warning('Unable to extract enum field value: %s', call)
        return None
    # Done, return the field
    return PBEnumField(field_name, False, field_number)


def _is_repeated_field(default_value):
    # type: (Union[astroid.Const, astroid.List]) -> bool
    """
    Infer from the default value whether a field is repeated
    or not?

    :param default_value: An `astroid.Node` representing a default value
    :return: `True` if field is `repeated`, `False` if not.
    """
    return isinstance(default_value, astroid.List)


def _extract_custom_field_value(field_name, classes, module):
    # type: (str, List[PBClass], astroid.Module) -> Optional[PBClass]
    """
    2nd Pass: Populate the custom Protobuf classes in the message fields

    :param field_name: Name of the field
    :param classes: List of `PBClass` objects
    :param module: Parent module object given to us by PyLint
    :return: Nothing
    """
    for node in module.body:
        # Filter out any non-assignment nodes
        if not isinstance(node, astroid.Assign):
            logging.debug('Node is not of type Assign: %s', node)
            continue
        # Get the target from the assignment node
        if not node.targets:
            logging.debug('Node target length is too short: %s', node)
            continue
        target_node = node.targets[0]
        if not isinstance(target_node, astroid.AssignAttr):
            logging.debug('Node target is not AssignAttr: %s', target_node)
            continue
        # Get the expression from the target
        expr_node = target_node.expr
        if not isinstance(expr_node, astroid.Subscript):
            logging.debug('Node target expr is not Subscript: %s', expr_node)
            continue
        # Get the slice from the expression
        slice_node = expr_node.slice
        if not isinstance(slice_node, astroid.Index):
            logging.debug('Node slice is not Index: %s', slice_node)
            continue
        # Only a constant value is expected inside the slice node
        if not isinstance(slice_node.value, astroid.Const):
            logging.debug('Slice node is not a Const: %s', slice_node.value)
            continue
        # Get the the value out of this (assigned class)
        expr_value = expr_node.value
        if not isinstance(expr_value, astroid.Attribute):
            logging.debug('Expr value is not an Attribute: %s', expr_value)
            continue
        expr_expr = expr_value.expr
        if not isinstance(expr_expr, astroid.Name):
            logging.debug('Expr expr is not a Name: %s', expr_expr)
            continue
        # Extract the field name present in the slice node
        slice_node_name = slice_node.value.value
        if not isinstance(slice_node_name, str):
            logging.warning("Slice name is not a str: %s", slice_node_name)
            continue
        if field_name != slice_node_name:
            logging.debug('Slice node name %s does '
                          'not match field name %s'
                          '', slice_node_name, field_name)
            continue
        # Extract the descriptor name for this field name
        field_desc_name = node.value.name
        # Get the PBClass object for this field from its descriptor
        return next(c for c in classes if c.desc_name == field_desc_name)
    # Unfortunately unable to determine the complex type
    logging.warning('Unable to determine the complex'
                    ' type for field: %s', field_name)
    return None


def _extract_message_field(classes, module, call):
    # type: (List[PBClass], astroid.Module, astroid.Call) -> Optional[PBField]
    """
    Extract a message field from the `FieldDescriptor` call
    :param classes: List of all `PBClass` objects
    :param module: Reference to `astroid.Module` object
    :param call: An `astroid.Call` belonging to `FieldDescriptor` call
    :return: PBField for the corresponding field
    """
    field_name = None
    field_pb_type = None
    field_default = None
    for keyword in call.keywords:
        if keyword.arg == 'name':
            field_name = getattr(keyword.value, 'value', None)
        if keyword.arg == 'type':
            field_pb_type = getattr(keyword.value, 'value', None)
        if keyword.arg == 'default_value':
            field_default = next(keyword.value.infer())
    if field_name is None:
        logging.debug("Unable to find field name: %s", call.keywords)
        return None
    if field_pb_type is None:
        logging.debug("Unable to find field type: %s", call.keywords)
        return None
    if field_default is None:
        logging.debug("Unable to find field default: %s", call.keywords)
        return None
    if field_pb_type in COMPLEX_FIELDS:
        # type: Optional[PBClass]
        field_value = _extract_custom_field_value(field_name, classes, module)
        field_type = PBFieldType.COMPLEX
    else:
        try:
            # type: Scalar
            field_value = FIELD_TYPES[field_pb_type]
        except KeyError:
            logging.debug("Unknown field type: %s", field_pb_type)
            return None
        field_type = PBFieldType.SCALAR
    # We're done inferring about this field.
    return PBMessageField(field_name,
                          _is_repeated_field(field_default),
                          field_type,
                          field_value)


def _extract_desc_fields(classes, module, parent_call, class_type):
    # type: (List[PBClass], astroid.Module, astroid.Call, PBClassType) -> List[PBField]
    """
    Extract fields present in the given a `astroid.Call` object
    for a `Descriptor` call

    :param classes: List of `PBClass` objects
    :param module: Reference to `astroid.Module` object
    :param parent_call: Creation of descriptor via `astroid.Call`
    :param class_type: Class type of the parent descriptor
    :return: Extracted fields present in the protobuf class
    """
    # The call to create a field object descriptor will have
    # `fields` for messages and `values` for enums.
    if class_type == PBClassType.MESSAGE:
        kw_name = 'fields'
    elif class_type == PBClassType.ENUM:
        kw_name = 'values'
    else:
        logging.warning("Unknown Class Type: %s", class_type)
        return []
    # Extract the values passed to `fields`/`value` keyword
    # in the parent call to create a descriptor.
    kw_val = next(kw for kw in parent_call.keywords if kw.arg == kw_name)
    if not isinstance(kw_val, astroid.Keyword):
        logging.warning('kw_val is not a Keyword object: %s', kw_val)
        return []
    if not isinstance(kw_val.value, astroid.List):
        logging.warning('Fields value is not a List: %s', kw_val.value)
        return []
    # Extract the list of calls for each keyword in the `kw_name` values
    calls = kw_val.value.elts
    fields = []
    for call in calls:
        if class_type == PBClassType.ENUM:
            # type: Optional[PBField]
            enum_field = _extract_enum_field(call)
            if enum_field:
                fields.append(enum_field)
        else:
            # This is a message field
            # type: Optional[PBField]
            msg_field = _extract_message_field(classes, module, call)
            if msg_field:
                fields.append(msg_field)
    # Done, return the extracted fields
    return fields


def _populate_fields(classes, module):
    # type: (List[PBClass], astroid.Module) -> None
    """
    Populate all the fields in the messages.

    :param classes: List of `PBClass` objects
    :param module: Reference to `astroid.Module` object
    :return: Nothing
    """
    # Module globals
    module_globals = module.globals
    # Enumerate through the classes
    for cls in classes:
        # Get the assignment for the descriptor
        try:
            # type: List[astroid.Assign]
            assign_line = module_globals[cls.desc_name]
        except KeyError:
            logging.error('Descriptor %s for class'
                          ' %s was not found', cls.desc_name, cls.name)
            continue
        # The assignment line should have at least one assignment
        if not assign_line:
            logging.warning('Descriptor assignment %s '
                            'has no nodes', cls.desc_name)
            continue
        # Get the assignment node itself
        assign_node = assign_line[0]
        if not isinstance(assign_node, astroid.AssignName):
            logging.warning('Descriptor assign_node is '
                            'not AssignName: %s', assign_node)
            continue
        # Parent should be an `astroid.Assign object
        if not isinstance(assign_node.parent, astroid.Assign):
            logging.warning('Descriptor assign_node parent '
                            'is not Assign: %s', assign_node.parent)
            continue
        call_node = assign_node.parent.value
        if not isinstance(call_node, astroid.Call):
            logging.warning('Assign\'s value is not a Call: %s', call_node)
            continue
        # Extract the fields
        fields = _extract_desc_fields(classes, module, call_node, cls.type)
        # Append it to the fields
        cls.fields.extend(fields)


def _transform(node):
    # type: (astroid.Module) -> astroid.Module
    """
    Callback function registered with PyLint to transform
    a particular node.

    :param node:  An `astroid.Module` node
    :return:
    """
    # Build a mapping of all the classes in the Protobuf module.
    # First identify the classes and its descriptors
    classes = _extract_classes(node.globals)
    # Populate the fields in these classes
    _populate_fields(classes, node)
    # Generate the import statements
    imports_str = '\n'.join('import ' + imp for imp in IMPORTS)
    # Generate the classes corresponding to protobuf messages & enums
    classes_str = '\n\n'.join(str(cls) for cls in classes)
    # Combine the above two to create code
    code = '\n\n'.join([imports_str, classes_str])
    # Copy some fields from the old node to the new node
    new_node = astroid.parse(code)
    new_node.name = node.name
    new_node.doc = node.doc
    new_node.file = node.file
    new_node.path = node.path
    new_node.package = node.package
    new_node.pure_python = node.pure_python
    new_node.parent = node.parent
    # Debug
    print(new_node.as_string())
    # Return the new node created by us
    return new_node


def _looks_like_pb2(node):
    # type: (astroid.Module) -> bool
    """
    Predicate function that determines when PyLint
    has to call our plugin's `_transform` callback on a node.

    :param node: An `astroid.Module` node
    :return: None
    """
    # Keep a list of ignored `_pb2` module names
    ignored = {
        'google.protobuf.descriptor_pb2'
    }
    # Filter out everything that doesn't end with `_pb2`
    return node.qname().endswith("_pb2") and node.qname() not in ignored


def configure_logging():
    # type: () -> None
    """
    Configure Logging for this plugin
    """
    if 'PYLINT_PROTOBUF_LOGLEVEL' in os.environ:
        level_str = os.environ['PYLINT_PROTOBUF_LOGLEVEL']
        try:
            level = getattr(logging, level_str)
        except AttributeError:
            logging.error('Unknown log level: %s', level_str)
            level = DEFAULT_LOG_LEVEL
    else:

        level = DEFAULT_LOG_LEVEL
    logging.basicConfig(level=level)


def register(_):
    # type: (Any) -> None
    """
    Register this plugin with the PyLint framework
    and perform any initialization needed by this plugin
    """


# Configure logging
configure_logging()

# Register the transformation function and the predicate
astroid.MANAGER.register_transform(astroid.nodes.Module,
                                   _transform,
                                   _looks_like_pb2)
