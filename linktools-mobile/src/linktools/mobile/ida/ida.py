import re


def get_point_size():
    return 8 if BADADDR == 0xFFFFFFFFFFFFFFFF else 4


def get_point_value(addr):
    return get_wide_dword(addr) if get_point_size() == 4 else get_qword(addr)


def get_func_name_ex(addr):
    func = ida_funcs.get_func(addr)
    if func:
        name = get_func_off_str(func.start_ea)
        if name is None or len(name) == 0:
            name = get_func_name(func.start_ea)
        return name
    return ""


def _fix_struc_member_name(member_name):
    if member_name is not None and member_name != "":
        member_name = re.sub(r"((?<=\)):[\s\S]*$)|(`virtual thunk to')|( )", "", member_name)
        member_name = member_name.replace(",", "::::")
        member_name = member_name.replace("<", "[")
        member_name = member_name.replace(">", "]")
        member_name = member_name.replace("=", "[equal]")
        member_name = member_name.replace("&", "[ref]")
        member_name = member_name.replace("*", "[ptr]")
        member_name = member_name.replace("~", "[destructor]")
    return member_name


def _is_bad_struc_id(struc_id):
    return struc_id == -1 or struc_id == BADADDR


def _get_or_add_struc(struc_name):
    struc_id = get_struc_id(struc_name)
    if _is_bad_struc_id(struc_id):
        struc_id = add_struc(-1, struc_name, 0)
    return struc_id


def _set_member_type(struc_id, member_type, member_offset):
    tinfo = idaapi.tinfo_t()
    struc = ida_struct.get_struc(struc_id)
    if not struc:
        return False
    member = ida_struct.get_member(struc, member_offset)
    if not member:
        return False
    idaapi.parse_decl2(idaapi.cvar.idati, '%s;' % member_type, tinfo, idaapi.PT_TYP)
    idaapi.set_member_tinfo2(struc, member, 0, tinfo, idaapi.SET_MEMTI_COMPATIBLE)
    return True


def _del_struc_member(struc_id):
    while del_struc_member(struc_id, get_struc_size(struc_id) - 1):
        pass
    return True


def _add_struc_member(struc_id, member_name, member_offset,
                      flag=FF_DWORD if get_point_size() == 4 else FF_QWORD, typeid=0, nbytes=get_point_size()):
    if _is_bad_struc_id(struc_id) \
            or member_name is None \
            or member_name == "" \
            or member_offset < 0:
        return False
    del_struc_member(struc_id, member_offset)

    i = 0
    member_name = name = _fix_struc_member_name(member_name)
    while get_member_offset(struc_id, member_name) != -1:
        member_name = "%s_%d" % (name, i)
        i = i + 1

    ret = add_struc_member(struc_id, member_name, member_offset, flag, typeid, nbytes)
    if ret < 0:
        print("error: add struct member %s failed (%d)" % (name, ret))
        return False
    return True


def add_vtbl_struc(vtbl_struc_name, vtbl_start, vtbl_end):
    vtbl_struc_id = _get_or_add_struc(vtbl_struc_name)
    if _is_bad_struc_id(vtbl_struc_id):
        print("error: add vtbl struct %s failed" % vtbl_struc_name)
        return False

    _del_struc_member(vtbl_struc_id)
    for addr in range(vtbl_start, vtbl_end, get_point_size()):
        result = False
        ea = get_point_value(addr)
        off = addr - vtbl_start
        if not result:
            result = _add_struc_member(vtbl_struc_id, get_func_name_ex(ea), off)
        if not result:
            result = _add_struc_member(vtbl_struc_id, "field_%x" % off, off)
        if result:
            set_member_cmt(vtbl_struc_id, off,
                           "addr: %xh\nname: %s" % (ea, get_func_name(ea)), 0)

    print("add vtbl struct %s finish" % vtbl_struc_name)
    return True


def add_vtbl_member(struc_name, vtbl_start, vtbl_end, vtbl_offset=0):
    struc_id = _get_or_add_struc(struc_name)
    vtbl_struc_name = "vtbl_%s_%d" % (struc_name, vtbl_offset)
    vtbl_struc_member = "vtbl_%d" % vtbl_offset
    if _is_bad_struc_id(struc_id):
        print("error: add struct %s failed" % vtbl_struc_name)
        return False

    if not add_vtbl_struc(vtbl_struc_name, vtbl_start, vtbl_end):
        return False
    if not _add_struc_member(struc_id, vtbl_struc_member, vtbl_offset):
        return False
    if not _set_member_type(struc_id, "%s *" % vtbl_struc_name, vtbl_offset):
        return False

    print("add vtbl member %s.%s finish" % (struc_name, vtbl_struc_member))
    return True


def set_user_reg(reg_name, reg_value, addr_start, addr_end):
    for addr in range(addr_start, addr_end):
        SetRegEx(addr, reg_name, reg_value, SR_user)


def _match_interface_begin(func_name):
    return func_name.find("onAsBinder") != -1


def _match_interface_end(func_name):
    return func_name.find("onTransact") != -1


def scan_interfaces(addr_start, addr_end, max_count=500):
    interfaces = []
    last_addr = BADADDR
    for addr in range(addr_start, addr_end, get_point_size()):
        func_name = get_func_name_ex(get_point_value(addr))
        if len(func_name) == 0:
            last_addr = BADADDR
        elif _match_interface_begin(func_name):
            last_addr = addr + get_point_size()
        elif last_addr == BADADDR:
            pass
        elif _match_interface_end(func_name):
            interfaces.append([last_addr, addr])
            last_addr = BADADDR
        elif addr_start + max_count * get_point_size() < last_addr:
            last_addr = BADADDR
    return interfaces


def _print_interfaces(inteface_start, inteface_end, simple=False):
    count = (inteface_end - inteface_start) / get_point_size()
    print("interfaces  start: 0x%x  end: 0x%x  count: %d" % (inteface_start, inteface_end, count))

    index = 0
    for addr in range(inteface_start, inteface_end, get_point_size()):
        index = index + 1
        func_addr = get_point_value(addr)
        if simple:
            print(get_func_name_ex(func_addr))
        else:
            print("(0x%02x) 0x%x: %s" % (index, func_addr, get_func_name_ex(func_addr)))
    print("")


def scan_all_interfaces(simple=True):
    interfaces = []
    addr_start = get_first_seg()
    while addr_start != BADADDR:
        addr_end = get_segm_end(addr_start)
        interfaces.extend(scan_interfaces(addr_start, addr_end))
        addr_start = get_next_seg(addr_start)
    for interface in interfaces:
        _print_interfaces(interface[0], interface[1], simple)
