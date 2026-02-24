/*************************************************************************
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2016-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "core.h"
#include "p2p_plugin.h"
#include "socket.h"
#include "ucx_plugin.h"

#define MAX_LINE_LEN 128
pthread_mutex_t nccl_bxi_lock = PTHREAD_MUTEX_INITIALIZER;

#define UCXCHECK(cmd)                                                          \
  do {                                                                         \
    int e = cmd;                                                               \
    if (UCS_OK != e) {                                                         \
      WARN("Failed: UCX error %s:%d '%d' %s\n", __FILE__, __LINE__, e,         \
           ucs_status_string(e));                                              \
      return ncclInternalError;                                                \
    }                                                                          \
  } while (0)

#define UCXCHECK_VOID(cmd)                                                     \
  do {                                                                         \
    int e = cmd;                                                               \
    if (UCS_OK != e) {                                                         \
      WARN("Failed: UCX error %s:%d '%d' %s\n", __FILE__, __LINE__, e,         \
           ucs_status_string(e));                                              \
    }                                                                          \
  } while (0)

struct oncewrap {
  pthread_once_t once;
};
static struct oncewrap onces[MAX_UCX_DEVS];

extern ncclDebugLogger_t pluginLogFunction;

int nccl_nb_bxi_devs = -1;
nccl_bxi_dev_t nccl_bxi_devs[MAX_UCX_DEVS];

static ncclResult_t nccl_ucx_bxi_devices(int *ndev) {
  *ndev = nccl_nb_bxi_devs;
  return ncclSuccess;
}

ncclResult_t nccl_ucx_bxi_get_properties(nccl_bxi_dev_t *devs, int dev,
                                         ncclNetProperties_t *props) {

  props->name = devs[dev].props.name;
  props->pciPath = devs[dev].props.pciPath;
  props->regIsGlobal = devs[dev].props.regIsGlobal;
  props->latency = devs[dev].props.latency;
  props->port = devs[dev].props.port_number;
  props->ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA;
  props->netDeviceType = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->guid = devs[dev].props.guid;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxCollBytes = devs[dev].props.max_coll_bytes;
  props->maxP2pBytes = devs[dev].props.max_p2p_bytes;
  props->maxRecvs = devs[dev].props.max_group_receives;
  props->maxComms = devs[dev].props.max_communicators;
  props->maxMultiRequestSize = 1;
  props->forceFlush = 1;

  return ncclSuccess;
}

ncclResult_t nccl_ucx_api_get_properties(int dev, ncclNetProperties_t *props) {
  return nccl_ucx_bxi_get_properties(nccl_bxi_devs, dev, props);
}

ncclResult_t
nccl_ucx_bxi_get_properties_v10(int dev, ncclNetProperties_v10_t *props_v10) {
  ncclNetProperties_t props = {0};
  ncclResult_t ret = nccl_ucx_bxi_get_properties(nccl_bxi_devs, dev, &props);
  if (ret != ncclSuccess)
    return ret;
  props_v10->name = props.name;
  props_v10->pciPath = props.pciPath;
  props_v10->guid = props.guid;
  props_v10->ptrSupport = props.ptrSupport;
  props_v10->regIsGlobal = props.regIsGlobal;
  props_v10->forceFlush = props.forceFlush;
  props_v10->speed = props.speed;
  props_v10->port = props.port;
  props_v10->latency = props.latency;
  props_v10->maxComms = props.maxComms;
  props_v10->maxRecvs = props.maxRecvs;
  props_v10->netDeviceType = props.netDeviceType;
  props_v10->netDeviceVersion = props.netDeviceVersion;
  props_v10->maxP2pBytes = props.maxP2pBytes;
  props_v10->maxCollBytes = props.maxCollBytes;
  props_v10->vProps.ndevs = props.vProps.ndevs;
  for (int i = 0; i < props.vProps.ndevs; i++) {
    props_v10->vProps.devs[i] = props.vProps.devs[i];
  }
  return ncclSuccess;
}

ncclResult_t nccl_ucx_bxi_get_properties_v9(int dev,
                                            ncclNetProperties_v9_t *props_v9) {
  ncclNetProperties_t props = {0};
  ncclResult_t ret = nccl_ucx_bxi_get_properties(nccl_bxi_devs, dev, &props);
  if (ret != ncclSuccess)
    return ret;
  props_v9->name = props.name;
  props_v9->pciPath = props.pciPath;
  props_v9->guid = props.guid;
  props_v9->ptrSupport = props.ptrSupport;
  props_v9->regIsGlobal = props.regIsGlobal;
  props_v9->forceFlush = props.forceFlush;
  props_v9->speed = props.speed;
  props_v9->port = props.port;
  props_v9->latency = props.latency;
  props_v9->maxComms = props.maxComms;
  props_v9->maxRecvs = props.maxRecvs;
  props_v9->netDeviceType = props.netDeviceType;
  props_v9->netDeviceVersion = props.netDeviceVersion;
  props_v9->maxP2pBytes = props.maxP2pBytes;
  props_v9->maxCollBytes = props.maxCollBytes;
  props_v9->vProps.ndevs = props.vProps.ndevs;
  for (int i = 0; i < props.vProps.ndevs; i++) {
    props_v9->vProps.devs[i] = props.vProps.devs[i];
  }
  return ncclSuccess;
}

ncclResult_t nccl_ucx_bxi_get_properties_v8(int dev,
                                            ncclNetProperties_v8_t *props_v8) {
  ncclNetProperties_t props = {0};
  ncclResult_t ret = nccl_ucx_bxi_get_properties(nccl_bxi_devs, dev, &props);
  if (ret != ncclSuccess)
    return ret;
  props_v8->name = props.name;
  props_v8->pciPath = props.pciPath;
  props_v8->guid = props.guid;
  props_v8->ptrSupport = props.ptrSupport;
  props_v8->regIsGlobal = props.regIsGlobal;
  props_v8->speed = props.speed;
  props_v8->latency = props.latency;
  props_v8->port = props.port;
  props_v8->maxComms = props.maxComms;
  props_v8->maxRecvs = props.maxRecvs;
  props_v8->netDeviceType = props.netDeviceType;
  props_v8->netDeviceVersion = props.netDeviceVersion;
  return ncclSuccess;
}

ncclResult_t nccl_ucx_bxi_get_properties_v7(int dev,
                                            ncclNetProperties_v7_t *props_v7) {
  ncclNetProperties_t props = {0};
  ncclResult_t ret = nccl_ucx_bxi_get_properties(nccl_bxi_devs, dev, &props);
  if (ret != ncclSuccess)
    return ret;
  props_v7->name = props.name;
  props_v7->pciPath = props.pciPath;
  props_v7->guid = props.guid;
  props_v7->ptrSupport = props.ptrSupport;
  props_v7->speed = props.speed;
  props_v7->latency = props.latency;
  props_v7->port = props.port;
  props_v7->maxComms = props.maxComms;
  props_v7->maxRecvs = props.maxRecvs;
  props_v7->netDeviceType = props.netDeviceType;
  props_v7->netDeviceVersion = props.netDeviceVersion;
  return ncclSuccess;
}

ncclResult_t nccl_ucx_bxi_get_properties_v6(int dev,
                                            ncclNetProperties_v6_t *props_v6) {
  ncclNetProperties_t props = {0};
  ncclResult_t ret = nccl_ucx_bxi_get_properties(nccl_bxi_devs, dev, &props);
  if (ret != ncclSuccess)
    return ret;
  props_v6->name = props.name;
  props_v6->pciPath = props.pciPath;
  props_v6->guid = props.guid;
  props_v6->ptrSupport = props.ptrSupport;
  props_v6->speed = props.speed;
  props_v6->latency = props.latency;
  props_v6->port = props.port;
  props_v6->maxComms = props.maxComms;
  props_v6->maxRecvs = props.maxRecvs;

  return ncclSuccess;
}

static int set_device_pci_path(nccl_bxi_dev_t *dev, const char *pci_slot_name) {
  int ret = ncclSuccess;
  char device_path[PATH_MAX];

  if (!pci_slot_name) {
    WARN("Invalid argument");
    return ncclInternalError;
  }

  unsigned domain_id, bus_id, device_id, function_id;
  if (sscanf(pci_slot_name, "%x:%x:%x.%x", &domain_id, &bus_id, &device_id,
             &function_id) != 4) {
    WARN("Failed to parse PCI slot name: %s", pci_slot_name);
    return ncclInternalError;
  }

  ret = snprintf(device_path, PATH_MAX,
                 "/sys/class/pci_bus/%04x:%02x/../../%04x:%02x:%02x.%01x",
                 domain_id, bus_id, domain_id, bus_id, device_id, function_id);
  if (ret < 0) {
    WARN("snprintf() failed\n");
    return ncclInternalError;
  }

  dev->props.pciPath = realpath(device_path, NULL);
  if (dev->props.pciPath == NULL) {
    WARN("pciPath: Could not find real path of %s", device_path);
    ret = ncclInternalError;
  }

  return ncclSuccess;
}

static ncclResult_t nccl_ucx_bxi_get_device_list(int *nb_bxi_devs_p,
                                                 nccl_bxi_dev_t *devices_p) {
  ncclResult_t ret = ncclSuccess;
  FILE *fp;
  char line[MAX_LINE_LEN];
  int ndevs = 0;

  // --- Step 1: get device names ---
  fp = popen("bxinic list", "r");
  if (!fp) {
    WARN("popen(bxinic list)");
    return ncclInternalError;
  }

  while (fgets(line, sizeof(line), fp) && ndevs < MAX_UCX_DEVS) {
    // strip newline
    line[strcspn(line, "\n")] = '\0';
    strncpy(devices_p[ndevs].props.name, line,
            sizeof(devices_p[ndevs].props.name));
    ndevs++;
  }
  pclose(fp);

  if (ndevs == 0) {
    WARN("No BXI devices found.");
    return ncclSuccess;
  }

  // --- Step 2: get UUIDs ---
  fp = popen("bxinic info uuid", "r");
  if (!fp) {
    WARN("popen(bxinic info uuid)");
    return ncclInternalError;
  }

  int idx = 0;
  while (fgets(line, sizeof(line), fp) && idx < ndevs) {
    line[strcspn(line, "\n")] = '\0';
    devices_p[idx].props.guid = strtoull(line, NULL, 16);
    idx++;
  }
  pclose(fp);

  if (idx != ndevs) {
    WARN("number of UUIDs (%d) != devices (%d)\n", idx, ndevs);
  }

  // --- Step 2: get UUIDs ---
  fp = popen("bxinic info pci_slot_name", "r");
  if (!fp) {
    WARN("popen(bxinic info uuid)");
    return ncclInternalError;
  }

  idx = 0;
  while (fgets(line, sizeof(line), fp) && idx < ndevs) {
    line[strcspn(line, "\n")] = '\0';
    ret = set_device_pci_path(&devices_p[idx], line);
    if (ret != ncclSuccess) {
      goto err;
    }
    idx++;
  }

  for (idx = 0; idx < ndevs; idx++) {
    devices_p[idx].props.port_number = 0;
    devices_p[idx].props.port_speed = 1024;
    devices_p[idx].props.latency = 1e-3;
    devices_p[idx].props.hmem_support = 0;
    devices_p[idx].props.dmabuf_support = 0;
    devices_p[idx].props.max_communicators = 1 << 18;
    devices_p[idx].props.max_group_receives = NCCL_NET_UCX_MAX_RECVS;
    devices_p[idx].props.rma_supported = 0;
    devices_p[idx].props.regIsGlobal = 0;
    devices_p[idx].props.max_write_inline_size = 256;
    devices_p[idx].props.max_mr_key_size = 256;
    devices_p[idx].props.max_p2p_bytes = 64 * 1024 * 1024;
    devices_p[idx].props.max_coll_bytes = 64 * 1024 * 1024;
  }

  *nb_bxi_devs_p = ndevs;

err:
  return ret;
}

ncclResult_t nccl_ucx_bxi_iface_init(int *nDevs, nccl_bxi_dev_t *nccl_bxi_devs,
                                     char *ncclIbIfName,
                                     union ncclSocketAddress *ncclIbIfAddr,
                                     ncclDebugLogger_t logFunction) {
  ncclResult_t ret = ncclSuccess;
  int nb_bxi_devs = *nDevs;
  pluginLogFunction = logFunction;

  if (nb_bxi_devs == -1) {
    for (int i = 0; i < MAX_UCX_DEVS; i++)
      onces[i].once = PTHREAD_ONCE_INIT;
    pthread_mutex_lock(&nccl_bxi_lock);
    if (nb_bxi_devs == -1) {
      int nIpIfs = 0;
      nb_bxi_devs = 0;
      NCCLCHECK(ncclFindInterfaces(ncclIbIfName, ncclIbIfAddr, MAX_IF_NAME_SIZE,
                                   1, &nIpIfs));
      if (nIpIfs != 1) {
        WARN("NET/BXI : No IP interface found.");
        ret = ncclInternalError;
        goto fail;
      }

      // Detect BXI cards
      int nbxi_devs = 0;

      if (ncclSuccess !=
          nccl_ucx_bxi_get_device_list(&nbxi_devs, nccl_bxi_devs)) {
        ret = ncclInternalError;
        goto fail;
      }

      for (int i = 0; i < nbxi_devs; i++) {

        TRACE(NCCL_NET, "NET/BXI: [%p] %s:%d speed=%d pciPath=%s", &devices[i],
              devices[i].props.name, devices[i].props.port_number,
              devices[i].props.port_speed, devices[i].props.pciPath);
      }
      if (nbxi_devs == 0) {
        INFO(NCCL_INIT | NCCL_NET, "NET/BXI : No device found.");
      }
      *nDevs = nbxi_devs;
      pthread_mutex_unlock(&nccl_bxi_lock);
    }
  }
exit:
  return ret;
fail:
  pthread_mutex_unlock(&nccl_bxi_lock);
  goto exit;
}

ncclResult_t nccl_ucx_bxi_init(void **ctx, uint64_t commId,
                               ncclNetCommConfig_v11_t *config,
                               ncclDebugLogger_t logFunction,
                               ncclProfilerCallback_t profFunction) {
  ncclResult_t ret;

  ret = nccl_ucx_param_init();
  if (ret != ncclSuccess) {
    return ret;
  }

  return nccl_ucx_bxi_iface_init(&nccl_nb_bxi_devs, nccl_bxi_devs, if_name,
                                 &nccl_ucx_if_addr, logFunction);
}

ncclResult_t nccl_ucx_bxi_init_v10(ncclDebugLogger_t logFunction,
                                   ncclProfilerCallback_t profilerCallback) {
  return nccl_ucx_bxi_init(NULL, 0, NULL, logFunction, NULL);
}

ncclResult_t nccl_ucx_bxi_connect_v10(int dev, ncclNetCommConfig_v10_t *config,
                                      void *opaqueHandle, void **sendComm,
                                      ncclNetDeviceHandle_v10_t **sendDevComm) {
  return nccl_ucx_connect(NULL, dev, opaqueHandle, sendComm, sendDevComm);
}

ncclResult_t nccl_ucx_bxi_listen_v10(int dev, void *opaqueHandle,
                                     void **listenComm) {
  return nccl_ucx_listen(NULL, dev, opaqueHandle, listenComm);
}

ncclResult_t nccl_ucx_bxi_connect_v9(int dev, void *opaqueHandle,
                                     void **sendComm,
                                     ncclNetDeviceHandle_t **sendDevComm) {
  return nccl_ucx_bxi_connect_v10(dev, NULL, opaqueHandle, sendComm,
                                  sendDevComm);
}

ncclResult_t nccl_ucx_bxi_init_v9(ncclDebugLogger_t logFunction) {
  return nccl_ucx_bxi_init_v10(logFunction, NULL);
}

ncclResult_t nccl_ucx_bxi_isend_v9(void *sendComm, void *data, size_t size,
                                   int tag, void *mhandle, void **request) {
  return nccl_ucx_isend(sendComm, data, size, tag, mhandle, NULL, request);
}

ncclResult_t nccl_ucx_bxi_irecv_v9(void *recvComm, int n, void **data,
                                   size_t *sizes, int *tags, void **mhandles,
                                   void **request) {
  return nccl_ucx_irecv(recvComm, n, data, sizes, tags, mhandles, NULL,
                        request);
}

ncclResult_t nccl_ucx_bxi_isend_v8(void *sendComm, void *data, int size,
                                   int tag, void *mhandle, void **request) {
  return nccl_ucx_bxi_isend_v9(sendComm, data, (size_t)size, tag, mhandle,
                               request);
}

ncclResult_t nccl_ucx_bxi_irecv_v8(void *recvComm, int n, void **data,
                                   int *sizes, int *tags, void **mhandles,
                                   void **request) {
  size_t sizes_sizet[NCCL_NET_IB_MAX_RECVS];
  for (int i = 0; i < n; i++)
    sizes_sizet[i] = sizes[i];
  return nccl_ucx_bxi_irecv_v9(recvComm, n, data, sizes_sizet, tags, mhandles,
                               request);
}

ncclResult_t nccl_ucx_bxi_regmr_v7(void *comm, void *data, int size, int type,
                                   void **mhandle) {
  return nccl_ucx_regmr(comm, data, (size_t)size, type, mhandle);
}

ncclResult_t nccl_ucx_bxi_connect_v6(int dev, void *handle, void **send_comm) {
  ncclNetDeviceHandle_v7_t *dev_handle = NULL;
  return nccl_ucx_bxi_connect_v9(dev, handle, send_comm, &dev_handle);
}

ncclResult_t nccl_ucx_bxi_accept_v6(void *listen_comm, void **recv_comm) {
  ncclNetDeviceHandle_v7_t *dev_handle = NULL;
  return nccl_ucx_accept(listen_comm, recv_comm, &dev_handle);
}

ncclNet_v11_t ucxBxiPlugin_v11 = {.name = "UCX_BXI",
                                  .init = nccl_ucx_bxi_init,
                                  .devices = nccl_ucx_bxi_devices,
                                  .getProperties = nccl_ucx_api_get_properties,
                                  .listen = nccl_ucx_listen,
                                  .connect = nccl_ucx_connect,
                                  .accept = nccl_ucx_accept,
                                  .regMr = nccl_ucx_regmr,
                                  .regMrDmaBuf = nccl_ucx_regmr_dmabuf,
                                  .deregMr = nccl_ucx_deregmr,
                                  .isend = nccl_ucx_isend,
                                  .irecv = nccl_ucx_irecv,
                                  .iflush = nccl_ucx_iflush,
                                  .test = nccl_ucx_test,
                                  .closeSend = nccl_ucx_close_send,
                                  .closeRecv = nccl_ucx_close_recv,
                                  .closeListen = nccl_ucx_close_listen,
                                  NULL /* getDeviceMr */,
                                  NULL /* irecvConsumed */,
                                  NULL,
                                  nccl_ucx_finalize,
                                  nccl_ucx_setNetAttr};

ncclNet_v10_t ucxBxiPlugin_v10 = {.name = "UCX_BXI",
                                  .init = nccl_ucx_bxi_init_v10,
                                  .devices = nccl_ucx_bxi_devices,
                                  .getProperties =
                                      nccl_ucx_bxi_get_properties_v10,
                                  .listen = nccl_ucx_bxi_listen_v10,
                                  .connect = nccl_ucx_bxi_connect_v10,
                                  .accept = nccl_ucx_accept,
                                  .regMr = nccl_ucx_regmr,
                                  .regMrDmaBuf = nccl_ucx_regmr_dmabuf,
                                  .deregMr = nccl_ucx_deregmr,
                                  .isend = nccl_ucx_isend,
                                  .irecv = nccl_ucx_irecv,
                                  .iflush = nccl_ucx_iflush,
                                  .test = nccl_ucx_test,
                                  .closeSend = nccl_ucx_close_send,
                                  .closeRecv = nccl_ucx_close_recv,
                                  .closeListen = nccl_ucx_close_listen,
                                  NULL /* getDeviceMr */,
                                  NULL /* irecvConsumed */,
                                  NULL};

ncclNet_v9_t ucxBxiPlugin_v9 = {.name = "UCX_BXI",
                                .init = nccl_ucx_bxi_init_v9,
                                .devices = nccl_ucx_bxi_devices,
                                .getProperties = nccl_ucx_bxi_get_properties_v9,
                                .listen = nccl_ucx_bxi_listen_v10,
                                .connect = nccl_ucx_bxi_connect_v9,
                                .accept = nccl_ucx_accept,
                                .regMr = nccl_ucx_regmr,
                                .regMrDmaBuf = nccl_ucx_regmr_dmabuf,
                                .deregMr = nccl_ucx_deregmr,
                                .isend = nccl_ucx_bxi_isend_v9,
                                .irecv = nccl_ucx_bxi_irecv_v9,
                                .iflush = nccl_ucx_iflush,
                                .test = nccl_ucx_test,
                                .closeSend = nccl_ucx_close_send,
                                .closeRecv = nccl_ucx_close_recv,
                                .closeListen = nccl_ucx_close_listen,
                                NULL /* getDeviceMr */,
                                NULL /* irecvConsumed */,
                                NULL};

ncclNet_v8_t ucxBxiPlugin_v8 = {
    .name = "UCX_BXI",
    .init = nccl_ucx_bxi_init_v9,
    .devices = nccl_ucx_bxi_devices,
    .getProperties = nccl_ucx_bxi_get_properties_v8,
    .listen = nccl_ucx_bxi_listen_v10,
    .connect = nccl_ucx_bxi_connect_v9,
    .accept = nccl_ucx_accept,
    .regMr = nccl_ucx_regmr,
    .regMrDmaBuf = nccl_ucx_regmr_dmabuf,
    .deregMr = nccl_ucx_deregmr,
    .isend = nccl_ucx_bxi_isend_v8,
    .irecv = nccl_ucx_bxi_irecv_v8,
    .iflush = nccl_ucx_iflush,
    .test = nccl_ucx_test,
    .closeSend = nccl_ucx_close_send,
    .closeRecv = nccl_ucx_close_recv,
    .closeListen = nccl_ucx_close_listen,
    NULL /* getDeviceMr */,
    NULL /* irecvConsumed */
};

ncclNet_v7_t ucxBxiPlugin_v7 = {
    .name = "UCX_BXI",
    .init = nccl_ucx_bxi_init_v9,
    .devices = nccl_ucx_bxi_devices,
    .getProperties = nccl_ucx_bxi_get_properties_v7,
    .listen = nccl_ucx_bxi_listen_v10,
    .connect = nccl_ucx_bxi_connect_v9,
    .accept = nccl_ucx_accept,
    .regMr = nccl_ucx_bxi_regmr_v7,
    .regMrDmaBuf = nccl_ucx_regmr_dmabuf,
    .deregMr = nccl_ucx_deregmr,
    .isend = nccl_ucx_bxi_isend_v8,
    .irecv = nccl_ucx_bxi_irecv_v8,
    .iflush = nccl_ucx_iflush,
    .test = nccl_ucx_test,
    .closeSend = nccl_ucx_close_send,
    .closeRecv = nccl_ucx_close_recv,
    .closeListen = nccl_ucx_close_listen,
    NULL /* getDeviceMr */,
    NULL /* irecvConsumed */
};

ncclNet_v6_t ucxBxiPlugin_v6 = {.name = "UCX_BXI",
                                .init = nccl_ucx_bxi_init_v9,
                                .devices = nccl_ucx_bxi_devices,
                                .getProperties = nccl_ucx_bxi_get_properties_v6,
                                .listen = nccl_ucx_bxi_listen_v10,
                                .connect = nccl_ucx_bxi_connect_v6,
                                .accept = nccl_ucx_bxi_accept_v6,
                                .regMr = nccl_ucx_bxi_regmr_v7,
                                .regMrDmaBuf = nccl_ucx_regmr_dmabuf,
                                .deregMr = nccl_ucx_deregmr,
                                .isend = nccl_ucx_bxi_isend_v8,
                                .irecv = nccl_ucx_bxi_irecv_v8,
                                .iflush = nccl_ucx_iflush,
                                .test = nccl_ucx_test,
                                .closeSend = nccl_ucx_close_send,
                                .closeRecv = nccl_ucx_close_recv,
                                .closeListen = nccl_ucx_close_listen};
